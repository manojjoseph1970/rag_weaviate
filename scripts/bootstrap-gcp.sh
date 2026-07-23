#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
CLUSTER_NAME="${CLUSTER_NAME:-rag-cluster}"
GAR_REPOSITORY="${GAR_REPOSITORY:-rag-images}"
NAMESPACE="${NAMESPACE:-rag}"
GSA_NAME="${GSA_NAME:-rag-app}"
KSA_NAME="${KSA_NAME:-rag-app}"

gcloud config set project "$PROJECT_ID"

gcloud services enable \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  pubsub.googleapis.com

if ! gcloud artifacts repositories describe "$GAR_REPOSITORY" \
  --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$GAR_REPOSITORY" \
    --repository-format=docker \
    --location="$REGION" \
    --description="RAG application images"
fi

if ! gcloud container clusters describe "$CLUSTER_NAME" \
  --region "$REGION" >/dev/null 2>&1; then
  gcloud container clusters create "$CLUSTER_NAME" \
    --region "$REGION" \
    --release-channel regular \
    --machine-type e2-standard-4 \
    --num-nodes 1 \
    --enable-ip-alias \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 4 \
    --disk-type pd-balanced \
    --disk-size 100
fi

gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID"

GSA_EMAIL="${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "$GSA_EMAIL" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$GSA_NAME" \
    --display-name="RAG application workload"
fi

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${GSA_EMAIL}" \
  --role="roles/pubsub.subscriber" \
  --condition=None >/dev/null

kubectl create namespace "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NAMESPACE" create serviceaccount "$KSA_NAME" \
  --dry-run=client -o yaml | kubectl apply -f -

gcloud iam service-accounts add-iam-policy-binding "$GSA_EMAIL" \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA_NAME}]" \
  >/dev/null

kubectl -n "$NAMESPACE" annotate serviceaccount "$KSA_NAME" \
  "iam.gke.io/gcp-service-account=${GSA_EMAIL}" \
  --overwrite

if ! gcloud pubsub topics describe document-ingestion >/dev/null 2>&1; then
  gcloud pubsub topics create document-ingestion
fi

if ! gcloud pubsub subscriptions describe document-ingestion-sub >/dev/null 2>&1; then
  gcloud pubsub subscriptions create document-ingestion-sub \
    --topic document-ingestion \
    --ack-deadline 60 \
    --min-retry-delay 10s \
    --max-retry-delay 600s
fi

echo "GKE and application identity are ready."
echo "Next: create the weaviate-auth and rag-app-secrets Kubernetes secrets."

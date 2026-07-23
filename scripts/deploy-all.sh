#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
CLUSTER_NAME="${CLUSTER_NAME:-rag-cluster}"
GAR_REPOSITORY="${GAR_REPOSITORY:-rag-images}"
NAMESPACE="${NAMESPACE:-rag}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE_REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPOSITORY}/rag-app"

gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE_REPOSITORY}:${IMAGE_TAG}" .
docker push "${IMAGE_REPOSITORY}:${IMAGE_TAG}"

helm repo add weaviate https://weaviate.github.io/weaviate-helm
helm repo update

helm upgrade --install weaviate weaviate/weaviate \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --values helm/weaviate-values.yaml \
  --atomic \
  --wait \
  --timeout 15m

helm upgrade --install rag-app helm/rag-app \
  --namespace "$NAMESPACE" \
  --values helm/rag-app/values.yaml \
  --set-string image.repository="$IMAGE_REPOSITORY" \
  --set-string image.tag="$IMAGE_TAG" \
  --set-string config.gcpProjectId="$PROJECT_ID" \
  --set-string serviceAccount.annotations."iam\.gke\.io/gcp-service-account"="rag-app@${PROJECT_ID}.iam.gserviceaccount.com" \
  --atomic \
  --wait \
  --timeout 15m

kubectl -n "$NAMESPACE" get pods,pvc,services

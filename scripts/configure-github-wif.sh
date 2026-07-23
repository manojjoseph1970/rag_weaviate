#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${GITHUB_OWNER:?Set GITHUB_OWNER}"
: "${GITHUB_REPOSITORY:?Set GITHUB_REPOSITORY}"

POOL_ID="${POOL_ID:-github-pool}"
PROVIDER_ID="${PROVIDER_ID:-github-provider}"
DEPLOY_SA="${DEPLOY_SA:-github-gke-deployer}"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
DEPLOY_SA_EMAIL="${DEPLOY_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam workload-identity-pools describe "$POOL_ID" \
  --location global --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL_ID" \
    --location global \
    --project "$PROJECT_ID" \
    --display-name "GitHub Actions"
fi

if ! gcloud iam workload-identity-pools providers describe "$PROVIDER_ID" \
  --workload-identity-pool "$POOL_ID" \
  --location global \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
    --workload-identity-pool "$POOL_ID" \
    --location global \
    --project "$PROJECT_ID" \
    --display-name "GitHub provider" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
    --attribute-condition "assertion.repository=='${GITHUB_OWNER}/${GITHUB_REPOSITORY}'"
fi

if ! gcloud iam service-accounts describe "$DEPLOY_SA_EMAIL" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$DEPLOY_SA" \
    --project "$PROJECT_ID" \
    --display-name "GitHub GKE deployer"
fi

for role in roles/artifactregistry.writer roles/container.developer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:${DEPLOY_SA_EMAIL}" \
    --role "$role" \
    --condition=None >/dev/null
done

gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA_EMAIL" \
  --project "$PROJECT_ID" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_OWNER}/${GITHUB_REPOSITORY}" \
  >/dev/null

echo
echo "Create these GitHub Actions secrets:"
echo "GCP_WORKLOAD_IDENTITY_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"
echo "GCP_SERVICE_ACCOUNT=${DEPLOY_SA_EMAIL}"

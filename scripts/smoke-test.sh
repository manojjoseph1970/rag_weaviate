#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"

curl --fail --silent --show-error "${BASE_URL}/health"
echo

curl --fail --silent --show-error \
  -X POST "${BASE_URL}/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "renewal-policy",
    "title": "Renewal Policy",
    "source": "renewal_policy.md",
    "department": "Customer Success",
    "text": "Enterprise renewals should be reviewed 90 days before renewal. High-risk accounts require executive alignment. Critical support issues should be resolved before commercial negotiation."
  }'
echo

curl --fail --silent --show-error \
  -X POST "${BASE_URL}/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"When are enterprise renewals reviewed?","limit":3}'
echo

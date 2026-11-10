# Cloudflare Workers AI

Route chat, embeddings, reranking and decision requests to [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/), including the [Clef](https://developers.cloudflare.com/workers-ai/models/clef/) decision models.

## Run

Create a Cloudflare API token with the **Workers AI** permission, then from this directory:

```sh
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_AUTH_TOKEN="your-api-token"
agentgateway -f config.yaml
```

Ask Clef a question through the OpenAI Decisions API:

```sh
curl http://localhost:4000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clef",
    "input": "My order arrived broken. Please refund me.",
    "questions": [{"type": "predicate", "name": "refund_requested", "instructions": "Is the customer asking for a refund?"}]
  }'
```

## Client APIs

| Client path | Workers AI endpoint | Notes |
| --- | --- | --- |
| `/v1/chat/completions` | `/ai/v1/chat/completions` | Streaming supported. |
| `/v1/messages` | `/ai/v1/chat/completions` | Translated from Anthropic Messages. |
| `/v1/responses` | `/ai/v1/responses` | Native for `@cf/openai/*` models; translated to Chat Completions for the others, which Workers AI does not serve on Responses. |
| `/v1/embeddings` | `/ai/v1/embeddings` | Workers AI does not report usage. |
| `/v1/rerank` | `/ai/run/{model}` | Translated from the Cohere shape; results are sorted by relevance. |
| `/v1/systemone` | `/ai/run/{model}` | Clef's native API. |
| `/v1/decisions` | `/ai/run/{model}` | Translated to System One. |

Models are addressed by their Workers AI id, such as `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. Clef also accepts its short name (`clef`, `clef-flash`, `clef-omni`), so `clef` and `@cf/cloudflare/clef` are equivalent.

Native endpoints wrap their responses in Cloudflare's `{"result": ..., "success": true}` envelope, and every endpoint reports errors in it. Agentgateway removes the envelope and returns errors in the client's API format.

## Kubernetes

[kubernetes.yaml](kubernetes.yaml) shows the same provider as an `AgentgatewayModel` (`provider: Cloudflare`) and as an `AgentgatewayBackend` (`ai.provider.cloudflare`). Both read the API token from a Secret named `cloudflare-token`, whose `Authorization` key holds the token.

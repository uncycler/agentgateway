use serde::{Deserialize, Serialize};

use crate::types::ResponseType;
use crate::{AIError, logged_response_parsing, types};

/// Workers AI rerankers (`@cf/baai/bge-reranker-*`) on `/ai/run/{model}`.
pub mod from_rerank {
	use super::*;

	#[derive(Serialize)]
	struct Request {
		query: String,
		contexts: Vec<Context>,
		#[serde(skip_serializing_if = "Option::is_none")]
		top_k: Option<u32>,
	}

	#[derive(Serialize)]
	struct Context {
		text: String,
	}

	/// The `result` of a rerank call, once the v4 envelope is removed.
	#[derive(Deserialize)]
	struct Response {
		response: Vec<Ranked>,
		#[serde(default)]
		meta: Option<serde_json::Map<String, serde_json::Value>>,
	}

	#[derive(Deserialize)]
	struct Ranked {
		id: u32,
		score: f64,
	}

	pub fn translate(req: &types::rerank::Request) -> Result<Vec<u8>, AIError> {
		if req.documents.is_empty() {
			return Err(AIError::MissingField("rerank documents".into()));
		}
		let body = Request {
			query: req.query.clone(),
			// The position in `contexts` is the `id` of each ranked result.
			contexts: req
				.documents
				.iter()
				.map(|d| Context { text: d.as_text() })
				.collect(),
			top_k: req.top_n,
		};
		serde_json::to_vec(&body).map_err(AIError::RequestMarshal)
	}

	/// Workers AI reports cost as numbered metric pairs; return the one named `input_tokens`.
	fn input_tokens(meta: &serde_json::Map<String, serde_json::Value>) -> Option<f64> {
		(1..=3).find_map(|i| {
			(meta.get(&format!("cost_metric_name_{i}"))?.as_str()? == "input_tokens")
				.then(|| meta.get(&format!("cost_metric_value_{i}"))?.as_f64())
				.flatten()
		})
	}

	/// Expects the unwrapped `result`. Ranked ids are the document positions; the response does not
	/// echo document text.
	pub fn translate_response(bytes: &[u8]) -> Result<Box<dyn ResponseType>, AIError> {
		let resp: Response = serde_json::from_slice(bytes).map_err(logged_response_parsing(bytes))?;
		let mut results: Vec<types::rerank::RerankResult> = resp
			.response
			.into_iter()
			.map(|r| types::rerank::RerankResult {
				index: r.id,
				relevance_score: r.score,
				document: None,
			})
			.collect();
		// Cohere orders results by relevance; Workers AI keeps the input order.
		results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
		let meta = resp
			.meta
			.as_ref()
			.and_then(input_tokens)
			.map(|input_tokens| types::rerank::Meta {
				tokens: Some(types::rerank::Tokens {
					input_tokens: Some(input_tokens),
					output_tokens: None,
				}),
				..Default::default()
			});
		Ok(Box::new(types::rerank::Response {
			id: None,
			results,
			meta,
			rest: serde_json::Value::Null,
		}))
	}

	#[cfg(test)]
	mod tests {
		use super::*;
		use crate::LogContentFields;

		#[test]
		fn request_uses_contexts() {
			let req: types::rerank::Request = serde_json::from_value(serde_json::json!({
				"model": "@cf/baai/bge-reranker-base",
				"query": "cat",
				"documents": ["a cat", {"text": "a car"}],
				"top_n": 1,
			}))
			.unwrap();
			let body: serde_json::Value = serde_json::from_slice(&translate(&req).unwrap()).unwrap();
			assert_eq!(
				body,
				serde_json::json!({
					"query": "cat",
					"contexts": [{"text": "a cat"}, {"text": "a car"}],
					"top_k": 1,
				})
			);
		}

		#[test]
		fn empty_documents_are_rejected() {
			let req: types::rerank::Request =
				serde_json::from_value(serde_json::json!({"query": "cat", "documents": []})).unwrap();
			assert!(translate(&req).is_err());
		}

		#[test]
		fn response_is_sorted_with_usage() {
			let body = br#"{"meta":{"cost_metric_name_1":"input_tokens","cost_metric_value_1":14,"cost_metric_name_2":null,"cost_metric_value_2":null,"neurons":0.004},"response":[{"id":0,"score":0.0007},{"id":1,"score":0.9994}]}"#;
			let resp = translate_response(body).unwrap();
			let out: serde_json::Value = serde_json::from_slice(&resp.serialize().unwrap()).unwrap();
			assert_eq!(out["results"][0]["index"], 1);
			assert_eq!(out["results"][1]["index"], 0);
			assert_eq!(
				resp
					.to_llm_response(LogContentFields::default())
					.input_tokens,
				Some(14)
			);
		}
	}
}

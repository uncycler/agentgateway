//! Cloudflare Workers AI.
//!
//! Chat, Responses and embeddings use the OpenAI-compatible `/ai/v1/*` endpoints, which reply
//! without an envelope. Models that only have a native API (Clef for System One, the rerankers)
//! are called on `/ai/run/{model}`, whose replies are wrapped in the Cloudflare v4 envelope
//! `{"result": ..., "success": true, "errors": [], "messages": []}`. Errors from every endpoint use
//! that envelope, with `success: false`.
//!
//! See <https://developers.cloudflare.com/workers-ai/>.

use agent_core::strng;
use agent_core::strng::Strng;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::{AIError, RouteType, apply, model_path, types};

#[apply(schema!)]
#[cfg_attr(feature = "schema", schemars(rename = "CloudflareProvider"))]
pub struct Provider {
	/// Model ID to send to Workers AI, overriding the model in the client request.
	#[serde(default, rename = "model", skip_serializing_if = "Option::is_none")]
	pub model_override: Option<Strng>,
	/// The Cloudflare account ID that owns the Workers AI usage.
	pub account_id: Strng,
}

impl super::Provider for Provider {
	const NAME: Strng = strng::literal!("cloudflare.workers_ai");
}

pub const DEFAULT_HOST_STR: &str = "api.cloudflare.com";
pub const DEFAULT_HOST: Strng = strng::literal!(DEFAULT_HOST_STR);

/// Namespace of Cloudflare's own models, such as Clef.
const CLOUDFLARE_MODEL_PREFIX: &str = "@cf/cloudflare/";

impl Provider {
	/// Base path of the account's Workers AI API.
	fn base_path(&self) -> Result<String, AIError> {
		if !model_path::is_safe_segment(&self.account_id) {
			return Err(AIError::UnsupportedConversion(strng::literal!(
				"cloudflare accountId must be a single path segment"
			)));
		}
		Ok(format!("/client/v4/accounts/{}/ai", self.account_id))
	}

	/// Whether the model serves the Responses API. Workers AI only implements it for OpenAI's
	/// open-weight models; the others reject it, so Responses clients are translated to Completions.
	pub fn is_openai_model(model: &str) -> bool {
		model.starts_with("@cf/openai/")
	}

	pub fn get_path_for_model(&self, route: RouteType, model: &str) -> Result<Strng, AIError> {
		let base = self.base_path()?;
		let suffix = match route {
			RouteType::SystemOne | RouteType::Rerank => {
				let model = run_model_id(route, model);
				if !model_path::is_safe_resource_name(&model) {
					return Err(AIError::UnsupportedConversion(strng::format!(
						"cloudflare model {model:?} is not a valid Workers AI model id"
					)));
				}
				return Ok(strng::format!("{base}/run/{model}"));
			},
			RouteType::Embeddings => "embeddings",
			RouteType::Responses => "responses",
			_ => "chat/completions",
		};
		Ok(strng::format!("{base}/v1/{suffix}"))
	}
}

/// The `/ai/run/{model}` id for a request model. Clef can be named without its namespace, as its
/// own API does (`clef`, `clef-flash`).
fn run_model_id(route: RouteType, model: &str) -> String {
	if route == RouteType::SystemOne && !model.starts_with('@') {
		format!("{CLOUDFLARE_MODEL_PREFIX}{model}")
	} else {
		model.to_string()
	}
}

/// Clef only accepts its short name (`clef`, `clef-flash`) in the request body, even though it is
/// addressed by its full id in the path.
pub fn systemone_body_model(model: &str) -> &str {
	model.strip_prefix(CLOUDFLARE_MODEL_PREFIX).unwrap_or(model)
}

/// Rewrite the `model` of a rendered System One body to the name Clef accepts.
pub fn finalize_systemone_body(body: Vec<u8>) -> Result<Vec<u8>, AIError> {
	let mut value: serde_json::Value =
		serde_json::from_slice(&body).map_err(AIError::RequestMarshal)?;
	if let Some(model) = value.get_mut("model")
		&& let Some(name) = model.as_str()
	{
		let short = systemone_body_model(name);
		if short != name {
			*model = serde_json::Value::String(short.to_string());
		}
	}
	serde_json::to_vec(&value).map_err(AIError::RequestMarshal)
}

#[derive(Deserialize)]
struct Envelope<'a> {
	#[serde(borrow, default)]
	result: Option<&'a RawValue>,
	success: Option<bool>,
	#[serde(default)]
	errors: Vec<EnvelopeMessage>,
}

#[derive(Deserialize)]
struct EnvelopeMessage {
	#[serde(default)]
	code: Option<serde_json::Value>,
	#[serde(default)]
	message: Option<String>,
}

fn parse_envelope(bytes: &[u8]) -> Option<Envelope<'_>> {
	let envelope: Envelope = serde_json::from_slice(bytes).ok()?;
	// Only the v4 envelope carries `success`; anything else is already a bare response.
	envelope.success.is_some().then_some(envelope)
}

/// Strip the v4 envelope from a successful `/ai/run` response. Bodies without an envelope are
/// returned unchanged.
pub fn unwrap_result(bytes: Bytes) -> Result<Bytes, AIError> {
	let Some(envelope) = parse_envelope(&bytes) else {
		return Ok(bytes);
	};
	match envelope.result {
		Some(result) if envelope.success == Some(true) => {
			Ok(Bytes::copy_from_slice(result.get().as_bytes()))
		},
		_ => Err(AIError::ResponseParsing(serde::de::Error::custom(format!(
			"cloudflare reported an unsuccessful response: {}",
			envelope_message(&envelope.errors)
		)))),
	}
}

fn envelope_message(errors: &[EnvelopeMessage]) -> String {
	let messages: Vec<&str> = errors.iter().filter_map(|e| e.message.as_deref()).collect();
	if messages.is_empty() {
		"unknown error".to_string()
	} else {
		messages.join("; ")
	}
}

fn openai_error_type(status: ::http::StatusCode) -> &'static str {
	match status.as_u16() {
		400 | 405 | 413 | 422 => "invalid_request_error",
		401 => "authentication_error",
		403 => "permission_error",
		404 => "not_found_error",
		429 => "rate_limit_error",
		_ => "api_error",
	}
}

/// Rewrite a Cloudflare v4 error envelope into the OpenAI error shape, which every client-facing
/// error translation starts from. Other bodies are returned unchanged.
pub fn normalize_error(bytes: &Bytes, status: ::http::StatusCode) -> Result<Bytes, AIError> {
	let Some(envelope) = parse_envelope(bytes) else {
		return Ok(bytes.clone());
	};
	if envelope.success != Some(false) {
		return Ok(bytes.clone());
	}
	let error = types::completions::typed::ChatCompletionErrorResponse {
		event_id: None,
		error: types::completions::typed::ChatCompletionError {
			r#type: Some(openai_error_type(status).to_string()),
			message: envelope_message(&envelope.errors),
			param: None,
			code: envelope.errors.iter().find_map(|e| e.code.clone()),
			event_id: None,
		},
	};
	Ok(Bytes::from(
		serde_json::to_vec(&error).map_err(AIError::ResponseMarshal)?,
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn provider() -> Provider {
		Provider {
			model_override: None,
			account_id: strng::literal!("0123456789abcdef0123456789abcdef"),
		}
	}

	#[rstest::rstest]
	#[case::completions(
		RouteType::Completions,
		"@cf/meta/llama-3.3-70b-instruct-fp8-fast",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1/chat/completions"
	)]
	#[case::messages_translated(
		RouteType::Messages,
		"@cf/meta/llama-3.3-70b-instruct-fp8-fast",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1/chat/completions"
	)]
	#[case::responses(
		RouteType::Responses,
		"@cf/openai/gpt-oss-20b",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1/responses"
	)]
	#[case::embeddings(
		RouteType::Embeddings,
		"@cf/baai/bge-m3",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1/embeddings"
	)]
	#[case::systemone_short(
		RouteType::SystemOne,
		"clef",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/run/@cf/cloudflare/clef"
	)]
	#[case::systemone_full(
		RouteType::SystemOne,
		"@cf/cloudflare/clef-flash",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/run/@cf/cloudflare/clef-flash"
	)]
	#[case::rerank(
		RouteType::Rerank,
		"@cf/baai/bge-reranker-base",
		"/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/run/@cf/baai/bge-reranker-base"
	)]
	fn paths(#[case] route: RouteType, #[case] model: &str, #[case] expected: &str) {
		assert_eq!(
			provider()
				.get_path_for_model(route, model)
				.unwrap()
				.as_str(),
			expected
		);
	}

	#[rstest::rstest]
	#[case::traversal("@cf/cloudflare/../../../v1/chat/completions")]
	#[case::query("clef?x=1")]
	#[case::empty("")]
	fn unsafe_run_models_are_rejected(#[case] model: &str) {
		assert!(
			provider()
				.get_path_for_model(RouteType::SystemOne, model)
				.is_err()
		);
	}

	#[test]
	fn unsafe_account_is_rejected() {
		let p = Provider {
			model_override: None,
			account_id: strng::literal!("abc/../def"),
		};
		assert!(p.get_path_for_model(RouteType::Completions, "m").is_err());
	}

	#[test]
	fn systemone_body_uses_short_model() {
		let body = br#"{"model":"@cf/cloudflare/clef","state":"x","questions":{}}"#.to_vec();
		let out = finalize_systemone_body(body).unwrap();
		let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
		assert_eq!(v["model"], "clef");
		assert_eq!(v["state"], "x");
		let body = br#"{"model":"clef-flash","state":"x","questions":{}}"#.to_vec();
		let v: serde_json::Value =
			serde_json::from_slice(&finalize_systemone_body(body).unwrap()).unwrap();
		assert_eq!(v["model"], "clef-flash");
	}

	#[test]
	fn unwraps_successful_envelope() {
		let body = Bytes::from_static(
			br#"{"result":{"model":"clef","answers":{"r":{"type":"noul","noul":0.99}},"usage":{"input_tokens":154,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
		);
		let out = unwrap_result(body).unwrap();
		assert_eq!(
			&out[..],
			br#"{"model":"clef","answers":{"r":{"type":"noul","noul":0.99}},"usage":{"input_tokens":154,"output_tokens":0}}"#
		);
	}

	#[test]
	fn bare_body_is_unchanged() {
		let body = Bytes::from_static(br#"{"model":"clef","answers":{}}"#);
		assert_eq!(unwrap_result(body.clone()).unwrap(), body);
	}

	#[test]
	fn unsuccessful_envelope_is_an_error() {
		let body = Bytes::from_static(
			br#"{"result":null,"success":false,"errors":[{"code":5006,"message":"bad"}],"messages":[]}"#,
		);
		assert!(unwrap_result(body).is_err());
	}

	#[rstest::rstest]
	#[case::no_model(
		::http::StatusCode::BAD_REQUEST,
		r#"{"errors":[{"message":"AiError: No such model","code":5007}],"success":false,"result":{},"messages":[]}"#,
		r#"{"event_id":null,"error":{"type":"invalid_request_error","message":"AiError: No such model","code":5007}}"#
	)]
	#[case::auth(
		::http::StatusCode::UNAUTHORIZED,
		r#"{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}],"messages":[]}"#,
		r#"{"event_id":null,"error":{"type":"authentication_error","message":"Authentication error","code":10000}}"#
	)]
	#[case::openai_shaped(
		::http::StatusCode::BAD_REQUEST,
		r#"{"error":{"type":"invalid_request_error","message":"x"}}"#,
		r#"{"error":{"type":"invalid_request_error","message":"x"}}"#
	)]
	#[case::not_json(::http::StatusCode::BAD_GATEWAY, "upstream down", "upstream down")]
	fn errors_are_normalized(
		#[case] status: ::http::StatusCode,
		#[case] input: &'static str,
		#[case] expected: &str,
	) {
		let out = normalize_error(&Bytes::from_static(input.as_bytes()), status).unwrap();
		let out: serde_json::Value = serde_json::from_slice(&out)
			.unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&out).into()));
		let expected: serde_json::Value = serde_json::from_str(expected)
			.unwrap_or_else(|_| serde_json::Value::String(expected.to_string()));
		assert_eq!(out, expected);
	}
}

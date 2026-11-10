use agent_core::strng;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{Value, json};

use crate::http::{Body, Response};
use crate::llm::{
	AIProvider, CacheTokenConvention, InputFormat, LLMRequest, Provider as _, ProviderState,
	RequestResult, RouteType, cloudflare, custom,
};
use crate::proxy::httpproxy::PolicyClient;
use crate::store::LLMResponsePolicies;
use crate::test_helpers::proxymock::setup_proxy_test;
use crate::types::agent::Target;

const ACCOUNT: &str = "0123456789abcdef0123456789abcdef";

fn provider(model: Option<&str>) -> AIProvider {
	AIProvider::Cloudflare(cloudflare::Provider {
		model_override: model.map(strng::new),
		account_id: strng::new(ACCOUNT),
	})
}

fn backend_info() -> crate::http::auth::BackendInfo {
	crate::http::auth::BackendInfo {
		target: crate::types::agent::BackendTarget::Invalid,
		call_target: Target::from((cloudflare::DEFAULT_HOST_STR, 443)),
		inputs: setup_proxy_test("{}").unwrap().pi,
	}
}

fn llm_request(input_format: InputFormat, model: &str) -> LLMRequest {
	LLMRequest {
		input_tokens: None,
		input_format,
		cache_convention: CacheTokenConvention::pending(),
		request_model: model.into(),
		original_model: None,
		provider: cloudflare::Provider::NAME,
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	}
}

fn json_request(path: &str, body: Value) -> crate::http::Request {
	::http::Request::builder()
		.uri(path)
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(serde_json::to_vec(&body).unwrap()))
		.unwrap()
}

fn upstream(status: ::http::StatusCode, body: &str) -> Response {
	::http::Response::builder()
		.status(status)
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(body.to_string()))
		.unwrap()
}

async fn respond(
	provider: &AIProvider,
	req: LLMRequest,
	resp: Response,
) -> (::http::StatusCode, Value) {
	let resp = provider
		.process_response(
			PolicyClient::new(setup_proxy_test("{}").unwrap().pi),
			req,
			LLMResponsePolicies::default(),
			None,
			Default::default(),
			None,
			resp,
		)
		.await
		.expect("response should be processed");
	let status = resp.status();
	let body = resp.into_body().collect().await.unwrap().to_bytes();
	(
		status,
		serde_json::from_slice(&body).expect("response should be JSON"),
	)
}

/// The upstream authority and path `setup_request` resolves for a route and request model. The
/// scheme comes from the connection's TLS policy, not from here.
fn upstream_url(route: RouteType, model: &str) -> String {
	let mut req = ::http::Request::builder()
		.uri("http://gateway/v1/anything")
		.body(Body::empty())
		.unwrap();
	let llm = llm_request(InputFormat::Completions, model);
	provider(None)
		.setup_request(&mut req, route, Some(&llm), None, None, false, None, None)
		.expect("setup_request should succeed");
	format!(
		"{}{}",
		req.uri().authority().unwrap(),
		req.uri().path_and_query().unwrap()
	)
}

#[rstest::rstest]
#[case::completions(
	RouteType::Completions,
	"@cf/meta/llama-3.3-70b-instruct-fp8-fast",
	"/ai/v1/chat/completions"
)]
#[case::responses(RouteType::Responses, "@cf/openai/gpt-oss-20b", "/ai/v1/responses")]
#[case::embeddings(RouteType::Embeddings, "@cf/baai/bge-m3", "/ai/v1/embeddings")]
#[case::systemone(RouteType::SystemOne, "clef", "/ai/run/@cf/cloudflare/clef")]
#[case::rerank(
	RouteType::Rerank,
	"@cf/baai/bge-reranker-base",
	"/ai/run/@cf/baai/bge-reranker-base"
)]
fn routes_resolve_to_workers_ai(
	#[case] route: RouteType,
	#[case] model: &str,
	#[case] suffix: &str,
) {
	assert_eq!(
		upstream_url(route, model),
		format!("api.cloudflare.com/client/v4/accounts/{ACCOUNT}{suffix}")
	);
}

#[rstest::rstest]
#[case::llama("@cf/meta/llama-3.3-70b-instruct-fp8-fast", false)]
#[case::gpt_oss("@cf/openai/gpt-oss-20b", true)]
fn responses_only_for_openai_models(#[case] model: &str, #[case] responses: bool) {
	let p = provider(None);
	assert_eq!(
		p.supports_format(custom::ProviderFormat::Responses, model),
		responses
	);
	// Decisions reach Clef through System One.
	assert_eq!(
		p.non_chat_provider_format_for(InputFormat::Decisions, model),
		Some(custom::ProviderFormat::SystemOne)
	);
}

#[tokio::test]
async fn decisions_render_systemone_with_short_model() {
	let p = provider(Some("@cf/cloudflare/clef"));
	let req = json_request(
		"/v1/decisions",
		json!({
			"model": "anything",
			"input": "My order arrived broken. Please refund me.",
			"questions": [{"type": "predicate", "name": "refund", "instructions": "Is this a refund request?"}]
		}),
	);
	let RequestResult::Success {
		llm_request,
		request,
		upstream_route_type,
	} = p
		.process_decisions_request(&backend_info(), None, req, false, &mut None)
		.await
		.expect("decisions should convert")
	else {
		panic!("expected forwarded request");
	};
	assert_eq!(upstream_route_type, RouteType::SystemOne);
	assert_eq!(llm_request.request_model.as_str(), "@cf/cloudflare/clef");
	assert!(matches!(
		llm_request.provider_state,
		Some(ProviderState::SystemOneDecisions(_))
	));
	let body: Value = serde_json::from_slice(&request.collect().await.unwrap().to_bytes()).unwrap();
	assert_eq!(body["model"], "clef");
	assert_eq!(body["questions"]["refund"]["type"], "noul");
}

#[tokio::test]
async fn decisions_response_is_unwrapped_and_translated() {
	let p = provider(None);
	let req = json_request(
		"/v1/decisions",
		json!({
			"model": "clef",
			"input": "Refund please",
			"questions": [{"type": "predicate", "name": "refund", "instructions": "Is this a refund request?"}]
		}),
	);
	let RequestResult::Success { llm_request, .. } = p
		.process_decisions_request(&backend_info(), None, req, false, &mut None)
		.await
		.unwrap()
	else {
		panic!("expected forwarded request");
	};
	let (status, body) = respond(
		&p,
		llm_request,
		upstream(
			::http::StatusCode::OK,
			r#"{"result":{"model":"clef","answers":{"refund":{"type":"noul","noul":0.99}},"usage":{"input_tokens":150,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
		),
	)
	.await;
	assert_eq!(status, ::http::StatusCode::OK);
	assert_eq!(body["answers"][0]["type"], "predicate");
	assert_eq!(body["answers"][0]["name"], "refund");
	assert_eq!(body["answers"][0]["probability"], 0.99);
	assert_eq!(body["usage"]["input_tokens"], 150);
}

#[tokio::test]
async fn systemone_response_is_unwrapped() {
	let (status, body) = respond(
		&provider(None),
		llm_request(InputFormat::SystemOne, "clef"),
		upstream(
			::http::StatusCode::OK,
			r#"{"result":{"model":"clef","answers":{"r":{"type":"noul","noul":0.5}},"usage":{"input_tokens":3,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
		),
	)
	.await;
	assert_eq!(status, ::http::StatusCode::OK);
	assert_eq!(
		body,
		json!({"model":"clef","answers":{"r":{"type":"noul","noul":0.5}},"usage":{"input_tokens":3,"output_tokens":0}})
	);
}

#[tokio::test]
async fn rerank_round_trips_through_run_api() {
	let p = provider(None);
	let req = json_request(
		"/v1/rerank",
		json!({"model": "@cf/baai/bge-reranker-base", "query": "cat", "documents": ["a car", "a cat"], "top_n": 2}),
	);
	let RequestResult::Success {
		llm_request,
		request,
		upstream_route_type,
	} = p
		.process_rerank_request(&backend_info(), None, req, false, &mut None)
		.await
		.unwrap()
	else {
		panic!("expected forwarded request");
	};
	assert_eq!(upstream_route_type, RouteType::Rerank);
	let body: Value = serde_json::from_slice(&request.collect().await.unwrap().to_bytes()).unwrap();
	assert_eq!(
		body,
		json!({"query": "cat", "contexts": [{"text": "a car"}, {"text": "a cat"}], "top_k": 2})
	);
	let (status, body) = respond(
		&p,
		llm_request,
		upstream(
			::http::StatusCode::OK,
			r#"{"result":{"meta":{"cost_metric_name_1":"input_tokens","cost_metric_value_1":21},"response":[{"id":0,"score":0.04},{"id":1,"score":0.99}]},"success":true,"errors":[],"messages":[]}"#,
		),
	)
	.await;
	assert_eq!(status, ::http::StatusCode::OK);
	assert_eq!(
		body["results"][0],
		json!({"index": 1, "relevance_score": 0.99})
	);
	assert_eq!(
		body["results"][1],
		json!({"index": 0, "relevance_score": 0.04})
	);
}

const NO_SUCH_MODEL: &str = r#"{"errors":[{"message":"AiError: No such model","code":5007}],"success":false,"result":{},"messages":[]}"#;

#[rstest::rstest]
#[case::completions(InputFormat::Completions, "/error/type", "invalid_request_error")]
#[case::messages(InputFormat::Messages, "/type", "error")]
#[case::responses(InputFormat::Responses, "/error/type", "invalid_request_error")]
#[case::embeddings(InputFormat::Embeddings, "/error/type", "invalid_request_error")]
#[case::systemone(InputFormat::SystemOne, "/error/type", "invalid_request_error")]
fn envelope_errors_reach_each_client_format(
	#[case] input_format: InputFormat,
	#[case] pointer: &str,
	#[case] expected: &str,
) {
	let req = llm_request(input_format, "@cf/meta/llama-3.3-70b-instruct-fp8-fast");
	let out = provider(None)
		.process_error(
			&req,
			::http::StatusCode::BAD_REQUEST,
			&Bytes::from_static(NO_SUCH_MODEL.as_bytes()),
			None,
		)
		.expect("error should translate");
	let body: Value = serde_json::from_slice(&out).unwrap();
	assert_eq!(
		body.pointer(pointer).and_then(Value::as_str),
		Some(expected)
	);
	assert_eq!(
		body.pointer("/error/message").and_then(Value::as_str),
		Some("AiError: No such model")
	);
}

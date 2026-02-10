
# llm.py visual

## Flow: Check mock → (mock result) OR (OpenAI call → extract text → parse JSON) → dict

```mermaid
flowchart TD
  A[route_audience / generate_content / evaluate_content] --> B{is_mock_mode?}

  B -- Yes --> M1[_mock_route / _mock_generate / _mock_evaluate]
  M1 --> R1[Return deterministic JSON dict]

  B -- No --> C[_call_llm_json]
  C --> D[get_client]
  D --> E{OPENAI_API_KEY set?}
  E -- No --> X[RuntimeError: client not available]
  E -- Yes --> F[OpenAI Responses API: responses.create]
  F --> G[_extract_text]
  G --> H[_parse_json]
  H --> R2[Return JSON dict]

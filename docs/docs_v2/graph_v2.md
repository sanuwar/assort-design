```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 40, "rankSpacing": 50}, "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "fontSize": "12px"}}}%%
flowchart TD
  subgraph Routing
    route_source{ML model available?}
    ml_router[ml_router]
    llm_router[llm_router]
    route_audience[route_audience]
  end
  subgraph Generation
    specialist_generate[specialist_generate]
    evaluate[evaluate]
  end
  subgraph Tools
    tool_citation[tool_citation]
    tool_risk[tool_risk]
    tool_gate[tool_gate]
  end
  subgraph Persistence
    persist_attempt[persist_attempt]
    decision{pass?}
    revise[revise]
    persist_results[persist_results]
  end
  route_source -->|yes| ml_router
  route_source -->|no| llm_router
  ml_router --> route_audience
  llm_router --> route_audience
  route_audience --> specialist_generate
  specialist_generate --> evaluate
  evaluate --> tool_citation
  tool_citation --> tool_risk
  tool_risk --> tool_gate
  tool_gate --> persist_attempt
  persist_attempt --> decision
  decision -->|revise| revise
  decision -->|persist_results| persist_results
  revise --> specialist_generate
  persist_results --> END
  classDef llm fill:#E8F1FF,stroke:#4C6FFF,stroke-width:1px,color:#1E2A5A;
  classDef ml fill:#E7FAF7,stroke:#20B2AA,stroke-width:1px,color:#0D3B3A;
  classDef tool fill:#FFF6E5,stroke:#F4A340,stroke-width:1px,color:#6A3B00;
  classDef persist fill:#E9F9EE,stroke:#3CB371,stroke-width:1px,color:#114B2F;
  classDef control fill:#F2F2F2,stroke:#888,stroke-width:1px,color:#333;
  class route_audience,specialist_generate,evaluate llm;
  class tool_citation,tool_risk,tool_gate tool;
  class persist_attempt,persist_results persist;
  class ml_router ml;
  class llm_router llm;
  class route_source,decision,revise control;
  subgraph Legend
    legend_ml[ML step]
    legend_llm[LLM step]
    legend_tool[Tool step]
    legend_persist[Persistence]
    legend_control[Decision/control]
  end
  class legend_ml ml;
  class legend_llm llm;
  class legend_tool tool;
  class legend_persist persist;
  class legend_control control;

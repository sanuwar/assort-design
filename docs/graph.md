```mermaid
flowchart TD
  route_audience[route_audience] --> specialist_generate[specialist_generate]
  specialist_generate --> evaluate
  evaluate --> persist_attempt
  persist_attempt -->|revise| revise
  persist_attempt -->|persist_results| persist_results
  revise --> specialist_generate
  persist_results --> END

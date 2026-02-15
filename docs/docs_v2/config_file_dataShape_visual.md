# Agent Profiles Data Structure

## YAML Configuration Schema

```
agent_profiles.yaml (top-level dict)
├── audiences (dict, required, non-empty)
│   ├── <audience_name_1> (dict)
│   │   ├── display_name (string, required)
│   │   ├── system_prompt (string, required)
│   │   ├── required_sections (list[string], required)
│   │   └── default_max_words (int > 0, required)
│   └── <audience_name_2> ...
│
├── routing (dict, required)
│   ├── auto_router_prompt (string, required)
│   └── low_confidence_threshold (number 0..1, required)
│   ├── ml_router_threshold (number 0..1, optional)
│   ├── ml_router_margin (number 0..1, optional)
│   └── ml_router_min_samples (int >= 1, optional)
│
├── evaluation (dict, required)
│   └── evaluator_prompt (string, required)
│
└── generation (dict, required)
    └── prompt (string, required)
```

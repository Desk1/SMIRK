This directory will be used to log significant design decisions of the project, to aid in the creation of the project report

For each significant design decision, we make a short markdown file named with the date and brief description: '2026-03-25-decision-directory-creation'. Each file answers three questions: what options did you consider, why did you choose the one you did, and what are the known trade-offs?

e.g
```
Decision: Use a decorator-based registry rather than a config-driven factory

Options considered:
- Config-driven: define model specs entirely in YAML, no Python registration
- Decorator-based: each backbone file registers itself with @register_model
- Dictionary in one file: single registry.py enumerates all models

Chosen: Decorator-based

Rationale: Config-driven would require duplicating Python type information
(function signatures) in YAML. Single dictionary requires importing all
backbones at once. Decorator-based allows lazy imports and keeps each
backbone's spec co-located with its implementation.

Trade-offs: Slightly more magic; requires convention that all backbone files
are imported at startup.
```
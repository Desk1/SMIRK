Decision: Design of smirk/models/registry.py with ModelSpec and decorator-based registration

Options considered:
- Monolithic registry: Single dictionary or function that enumerates all model configurations
- Subclass-based approach: Each backbone creates a model subclass with overridden properties
- Decorator with metadata wrapper: Use @register_model decorator that attaches _model_spec with ModelSpec container

Chosen: Decorator with ModelSpec container

Rationale: The ModelSpec approach centralizes model metadata (resolution, mean, std, loader, expert_wrapper) in a single, strongly-typed container that each backbone provides. This eliminates the need for subclasses while keeping architecture-specific metadata organized in an extra dict. The @register_model decorator provides several benefits:
- **Early error detection**: Duplicate names raise immediately at import time, catching accidental double-imports before they cause runtime confusion
- **Introspection**: The _model_spec attribute stapled onto the decorated function allows callers to inspect model configuration programmatically
- **Self-contained backbones**: Each backbone file defines _build_expert_model internally and passes it via expert_wrapper=, keeping implementation details co-located with their specifications
- **Simple API**: Callers use get_expert_model("efficientnet_b0", num_experts=3, device=device) without manually reconstructing expert models

Trade-offs:
- Decorator registration requires a naming convention and disciplined imports to ensure all backbone files are loaded at startup
- The extra dict for architecture-specific metadata (weight paths, num_classes, outputs_tuple flag) requires documentation to prevent misuse
- Slight runtime overhead from decorator invocation, though negligible in practice since registration occurs once at import time

Key design properties:
- ModelSpec is reusable and extensible for future metadata requirements
- Backbone files are independent and self-contained
- Registry prevents accidental name collisions through immediate validation
- No circular import issues since expert_wrapper accepts the builder function, not the built model

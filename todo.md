- refactor smirk/data/sampler.py, move file handling etc into smirk/scrips/sample.py to be separate from actual sampling logic

- make smirk.models.registry.get_model() load weights by default unless weight location not specified or if weight location specified but not found (raise warning)

- integrate hydra config handling with scripts properly

- write tests for full data pipeline
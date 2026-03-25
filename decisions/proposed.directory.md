smirk/
├── configs/                    # All configuration lives here
│   ├── base.yaml               # Shared defaults
│   ├── datasets/
│   │   ├── celeba.yaml
│   │   └── ffhq.yaml
│   ├── models/
│   │   ├── resnet50.yaml
│   │   ├── inception_resnetv1.yaml
│   │   └── ...
│   ├── attacks/
│   │   ├── smile.yaml
│   │   ├── mirror_b.yaml
│   │   └── ppa.yaml
│   └── experiments/            # Full experiment configs composing the above
│       └── smile_casia_to_vggface2.yaml
│
├── smirk/                      # Main Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py        # GAN wrapper (StyleGAN interface)
│   │   ├── sampler.py          # Latent space sampling (replaces my_sample_z_w_space.py)
│   │   └── dataset.py          # CustomDataset, query dataset building
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py         # Central model registry - replaces all if/elif chains
│   │   ├── target.py           # Target model loading/wrapping
│   │   ├── surrogate.py        # Surrogate model + expert ensemble logic
│   │   └── backbones/          # Thin wrappers around each architecture
│   │       ├── resnet50.py
│   │       ├── inception_resnetv1.py
│   │       └── ...
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── long_tail.py        # Long-tailed weighting (build_weight_k, etc.)
│   │   ├── losses.py           # KL, CE, diversity losses as named functions
│   │   └── trainer.py          # Surrogate training loop
│   │
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class for all attacks
│   │   ├── population.py       # VectorizedPopulation classes
│   │   ├── whitebox.py         # Mirror-w, PPA, ours-w
│   │   └── blackbox.py         # Mirror-b, ours-current_maximum, ours-optimal_fit, ours-surrogate_model
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # top-1/top-k accuracy, L2, KNN
│   │   └── visualisation.py    # Image annotation, saving grids
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image.py            # crop_and_resize, normalize, resize_img
│       ├── latent.py           # clip_quantile_bound, p-space transforms
│       └── logging.py          # Structured logger, experiment dir management
│
├── scripts/                    # Thin entry-point scripts, no logic
│   ├── sample_latents.py
│   ├── build_query_dataset.py
│   ├── merge_logits.py
│   ├── train_surrogate.py
│   └── run_attack.py
│
├── tests/
│   ├── unit/
│   │   ├── test_long_tail.py
│   │   ├── test_losses.py
│   │   ├── test_image_utils.py
│   │   └── test_registry.py
│   └── integration/
│       └── test_pipeline_smoke.py
│
├── notebooks/                  # Exploration and result visualisation
│   └── results_analysis.ipynb
│
├── pyproject.toml
└── README.md
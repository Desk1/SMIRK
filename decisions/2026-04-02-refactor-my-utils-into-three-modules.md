Decision: Refactor my_utils.py into three separate utility modules

Options considered:
- Maintain monolithic my_utils.py: Keep all utilities in a single file
- Split by function type: Separate into three focused modules based on functionality domains
- Create a utils package: Establish a utils/ package with multiple submodules

Chosen: Split into three separate modules by functionality domain

Rationale: The current my_utils.py contains functions spanning three distinct concern areas:
1. **Image processing** (denormalize, normalize, crop_img, resize_img, clip, crop_and_resize) - handles image normalization, cropping, and resizing for different architectures
2. **Latent space handling** (compute_topk_labels, find_most_overlapped_topk, my_select_ind) - manages selection and analysis of latent codes and confidence scores
3. **Filesystem and path resolving** (create_folder, Tee) - handles directory creation and output redirection

Separating these concerns improves:
- **Maintainability**: Each module has a single responsibility, making code easier to understand and modify
- **Reusability**: Clients can import only what they need without loading unrelated utilities
- **Testability**: Focused modules enable more targeted unit testing
- **Readability**: Smaller, focused files are easier to navigate

The refactoring will also exclude unused functions (powerset, crop_img_for_ccs19ami raise AssertionError) from the new modules, cleaning up technical debt.

Trade-offs: 
- Minor increase in file count (3 new files) requires clients to update import statements
- Need to maintain consistency across the three modules regarding style and conventions
- Risk of circular imports if filesystem utilities need to reference image processing utilities (mitigated through careful dependency analysis)

Refactored modules:
- `image_utils.py`: Image processing functions with architecture-specific normalization and cropping
- `latent_utils.py`: Latent space analysis and label selection utilities
- `filesystem_utils.py`: Directory management and output utilities

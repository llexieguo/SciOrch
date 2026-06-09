# Third-Party Notices

This repository vendors third-party source code (see `ms-swift` below) and
includes derivative implementations inspired by and partially adapted from the
projects listed here.

## ms-swift (vendored)
- Source project: `ms-swift` (ModelScope), https://github.com/modelscope/ms-swift
- License: Apache License 2.0
- Copyright (c) ModelScope Contributors
- Usage: The entire `swift/` directory is a vendored copy of ms-swift, used
  largely unmodified as the offline-REINFORCE training framework. Original
  per-file copyright headers are retained; the Apache-2.0 license text is
  included at `LICENSES/Apache-2.0.txt`.

This repository also includes derivative implementations inspired by and
partially adapted from:

## AOrchestra
- Source project: `AOrchestra`
- License: Apache License 2.0
- Derived components:
  - Main-agent orchestration structure
  - Delegate/submit tool split
  - Memory pattern and usage/cost tracking ideas

## SGI-Bench
- Source project: `SGI-Bench`
- License: MIT
- Derived components:
  - Reasoning scoring prompt and MCA extraction logic
  - Reasoning-compatible output log structure

## License files
- Apache-2.0 text: `LICENSES/Apache-2.0.txt`
- SGI-Bench MIT text: `LICENSES/MIT-SGI-Bench.txt`

## Modification statement
Files with direct derivative logic include a header noting source project and that the implementation was modified for this project.

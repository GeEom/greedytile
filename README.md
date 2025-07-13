# GreedyTile

[![Crates.io](https://img.shields.io/crates/v/greedytile.svg)](https://crates.io/crates/greedytile)
[![CI](https://github.com/GeEom/greedytile/actions/workflows/ci.yml/badge.svg)](https://github.com/GeEom/greedytile/actions/workflows/ci.yml)
[![Rust](https://github.com/GeEom/greedytile/actions/workflows/rust.yml/badge.svg)](https://github.com/GeEom/greedytile/actions/workflows/rust.yml)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.txt)
[![codecov](https://codecov.io/gh/GeEom/greedytile/branch/master/graph/badge.svg)](https://codecov.io/gh/GeEom/greedytile)

Fast pattern synthesis preserving local tile constraints. Safe Rust implementation with CLI and library interfaces.

## Examples

| Source | Generated | Visualized |
|--------|-----------|------------|
| ![Source A](data/examples/a_4x.png) | ![Result A](data/examples/a_result_4x.png) | ![Vis A](data/examples/a_visualization_4x.gif) |
| ![Source B](data/examples/b_4x.png) | ![Result B](data/examples/b_result_4x.png) | ![Vis B](data/examples/b_visualization_4x.gif) |
| ![Source C](data/examples/c_4x.png) | ![Result C](data/examples/c_result_4x.png) | ![Vis C](data/examples/c_visualization_4x.gif) |
| ![Source D](data/examples/d_4x.png) | ![Result D](data/examples/d_result_4x.png) | ![Vis D](data/examples/d_visualization_4x.gif) |
| ![Source E](data/examples/e_4x.png) | ![Result E](data/examples/e_result_4x.png) | ![Vis E](data/examples/e_visualization_4x.gif) |
| ![Source F](data/examples/f_4x.png) | ![Result F](data/examples/f_result_4x.png) | ![Vis F](data/examples/f_visualization_4x.gif) |
| ![Source G](data/examples/g_4x.png) | ![Result G](data/examples/g_result_4x.png) | ![Vis G](data/examples/g_visualization_4x.gif) |

## Installation and usage

```bash
# Install with Rust 1.88.0 or later
cargo install greedytile

# Generate pattern from source image
greedytile input.png

# Process directory of images including rotation, mirroring, and animation
greedytile ./patterns/ -r -m -v
```
## Options


| Option | Description | Default | Without Option | With Option |
|----------|-------------|---------|----------------|-------------|
| `--seed` | Seed reproducible generation | 42 | ![Default Seed](data/examples/seed_default_4x.png) | ![Custom Seed](data/examples/seed_custom_4x.png) |
| `--iterations` | Maximum iterations| 1000 | ![Default Iterations](data/examples/iterations_default_4x.png) | ![More Iterations](data/examples/iterations_custom_4x.png) |
| `--prefill` | Use content of `<input>_pre.png` | disabled | ![No Prefill](data/examples/prefill_off_4x.png) | ![Prefill Enabled](data/examples/prefill_on_4x.png) |
| `--visualize`  | Generate placement animation | disabled | | ![Tile Placement Animation](data/examples/visualization_4x.gif) |
| `--width` | Maximum pixel width | unbounded | ![Unbounded](data/examples/size_unbounded_4x.png) | ![Bounded](data/examples/size_32x32_4x.png) |
| `--height` | Maximum pixel height | unbounded | | |
| `--rotate` | Enable 90°/180°/270° tile rotations | disabled | ![No Rotation](data/examples/rotate_off_4x.png) | ![Rotation Enabled](data/examples/rotate_on_4x.png) |
| `--mirror`| Enable tile reflection | disabled | ![No Mirror](data/examples/mirror_off_4x.png) | ![Mirror Enabled](data/examples/mirror_on_4x.png) |
| `--quiet` | Suppress progress output | verbose |
| `--no-skip` | Process overwriting existing output | skip existing |

## Details

GreedyTile generates patterns by placing 3×3 tiles extracted from a source image. Weights for randomly selecting placement are influenced by several factors:

1. **Entropy weighting**: Prioritizes positions where fewer tile options are valid (similar to [WFC](https://github.com/mxgmn/WaveFunctionCollapse/))
2. **Global balance correction**: If pixels are under-represented then selection is biased towards the source balance
3. **Distance-based probabilities**: Pixel distance patterns in the source are replicated in the output
4. **Deadlock recovery**: If no choices exist which respect tiles, pixels in a local area are removed before continuing

## Commentary

For efficiency, the global balance correction uses a normal distribution to approximate what should technically be a binomial distribution. The approximation becomes accurate at large counts where it matters most.

The distance probability weighting has more impact on easily tiled patterns than complex ones. Restrictive patterns often have less valid tile choices, making the probabilistic selection less relevant.

## Limitations

- **Fixed tile size**: Currently hardcoded to sample 3×3 pixel tiles from the source
- **Extensibility**: Would be improved by a plugin interface for adding new probability rules

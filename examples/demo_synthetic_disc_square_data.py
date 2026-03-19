from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_conv_attention import generate_disc_square_image, make_disc_square_types


def main() -> None:
    specs = make_disc_square_types()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, spec in zip(axes.flat, specs):
        image = generate_disc_square_image(spec)[0]
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"r={spec.radius}, s={spec.square_size}, near={spec.near_density}, far={spec.far_density}",
            fontsize=9,
        )
        axis.axis("off")
    fig.tight_layout()
    output_path = PROJECT_ROOT / "examples" / "demo_synthetic_disc_square_data.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

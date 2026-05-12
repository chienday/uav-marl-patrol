"""
Export utilities — Save artifacts to .tar.gz archives.
"""

import os
import tarfile
from datetime import datetime


def save_artifacts(
    source_dir: str,
    output_path: str = None,
    prefix: str = "uav_marl",
):
    """Save project artifacts (checkpoints, logs) to a .tar.gz archive."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{prefix}_{timestamp}.tar.gz"

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    print(f"Artifacts saved to: {output_path}")
    return output_path

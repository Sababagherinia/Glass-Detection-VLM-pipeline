# visulalization
import matplotlib.pyplot as plt
import pyoctomap
import numpy as np
from pathlib import Path

# Resolve absolute .bt path next to this script
bt_path = Path(__file__).with_name("my_map.bt")
if not bt_path.exists():
    raise FileNotFoundError(f"BT file not found: {bt_path}")

def load_octree(bt_file: Path) -> pyoctomap.OcTree:
    # Try OcTree.readBinary with str, then bytes
    tree = pyoctomap.OcTree(0.1)
    try:
        ok = tree.readBinary(str(bt_file))
    except TypeError:
        ok = tree.readBinary(str(bt_file).encode("utf-8"))
    if ok:
        return tree
    # Fallback: AbstractOcTree.read with bytes, then str
    try:
        at = pyoctomap.AbstractOcTree.read(str(bt_file).encode("utf-8"))
    except TypeError:
        at = pyoctomap.AbstractOcTree.read(str(bt_file))
    if at is None:
        raise RuntimeError(f"Failed to read octree from {bt_file}")
    # In most builds this is already an OcTree
    if isinstance(at, pyoctomap.OcTree):
        return at
    # Some builds expose cast; if not present, assume OcTree
    try:
        return pyoctomap.castToOcTree(at)  # type: ignore[attr-defined]
    except Exception:
        return at  # type: ignore[return-value]

# Load octree
tree = load_octree(bt_path)

# Collect occupied voxel centers
occupied_points = []
it = tree.begin_leafs()
end = tree.end_leafs()
while it != end:
    if tree.isNodeOccupied(it):
        p = it.getCoordinate()
        occupied_points.append([p.x(), p.y(), p.z()])
    it.next()

occupied_points = np.array(occupied_points, dtype=float)
if occupied_points.size == 0:
    print(f"No occupied nodes found in {bt_path.name}")
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2], c="r", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Occupied voxels")
    plt.show()

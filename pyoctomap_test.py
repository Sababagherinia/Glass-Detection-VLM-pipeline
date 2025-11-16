import pyoctomap
import numpy as np

# Create an octree with 0.1m resolution
tree = pyoctomap.OcTree(0.1)

# Add occupied points
tree.updateNode([1.0, 2.0, 3.0], True)
tree.updateNode([1.1, 2.1, 3.1], True)

# Add free space
tree.updateNode([0.5, 0.5, 0.5], False)

# Check occupancy
node = tree.search([1.0, 2.0, 3.0])
if node and tree.isNodeOccupied(node):
    print("Point is occupied!")

# Save to file
tree.write("my_map.bt")
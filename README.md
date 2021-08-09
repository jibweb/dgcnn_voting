# Addressing the Sim2Real gap in 3D object classification

## Installation

Requires PCL 1.7+

```
mkdir build
cd build
cmake ..
make
```

## Usage

To create the graph of object parts, use the utility function:

```
from preprocessing import get_graph_preprocessing_fn, p

preprocess = get_graph_preprocessing_fn(p)
nodes_feats, adj_mat, edge_feats, valid_indices = preprocess("path/to/mesh.ply")
```

It should be noted that the program only accept ply files including normals (vertex with properties x,y,z,nx,ny,nz and face with property vertex_indices). If you want to load other mesh, use your favorite libraries and modify the preprocessing file to load the mesh using: 

```
gc.initialize_mesh_from_array(vertices, triangles, normals)
```
OR 
```
gc.initialize_mesh_from_array(vertices, triangles)
```

Note that this will be a bit slower
PyDSG
=====




# Technical Notes

## Why is the edge information stored so weirdly?

One requirement for the library is that every layer needs to be useful
individually, outside of the context of the full scene graph data structure. In
oder to support this, the connectivity information for nodes in the same layer
(i.e., the sibling structure) needs to be stored as part of the layer. For
consistency, the parent/child information is also stored as part of each layer,
even though it might seem more straightforward to store this information at the
DSG-level instead. The PyDSG object can still look up node parent/child
connections, but it searches through each of the layers to find the appropriate
dictionary. Maybe this can be cleaned up eventually, but it works well enough
for now.

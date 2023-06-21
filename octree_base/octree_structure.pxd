# distutils: language=c++
#cython: language_level=3
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fmax, fmin
from libcpp.vector cimport vector
from libcpp.queue cimport queue, priority_queue
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp cimport bool

cimport octree_defs
# import octree_defs
from octree_defs cimport OTNode, initialiseOTNodePointer, PyOTNode, Octree, initialiseOctreePointer, PyOctree, NodeDist, NodeDistQueue

cdef void serialise_tree(Octree* octreePtr, OTNode* nodePtr, vector[int]* dataPtr)

cdef int buildTree_from_data(Octree* octreePtr, vector[int]* dataPtr, int dataIdx, OTNode* nodePtr, int depth, Py_ssize_t childIndex, 
        float length, float top_pos_x, float top_pos_y, float top_pos_z, OTNode* parentPtr, bool is2D)

cdef int extend_chilren_from_data(Octree* octreePtr, OTNode* parentPtr, vector[int]* dataPtr, int dataIdx)

cdef int extend_tree(Octree* octreePtr, int depth, float[:,:] points)

cdef void setNonSurfLeafs(Octree* octreePtr)

cdef void setNonSurfacePtrs(Octree* octreePtr, OTNode* nodePtr, int depth)

cdef void extendNearSurfLeaves(Octree* octreePtr, OTNode* nodePtr, int max_depth, float[:,:] points)

cdef int extend_chilren(Octree* octreePtr, OTNode* parentPtr, int max_depth, float[:,:] points)

cdef inline int checkPointInBounds(float px, float py, float pz, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z) nogil

cdef int buildTree(Octree* octreePtr, OTNode* nodePtr, int max_depth, int depth, Py_ssize_t childIndex, 
        float[:,:] points, float length, 
        float top_pos_x, float top_pos_y, float top_pos_z, OTNode* parentPtr, bool is2D)

cdef vector[Py_ssize_t] code2index3d(vector[Py_ssize_t] code) nogil

cdef vector[Py_ssize_t] index3d2code(vector[Py_ssize_t] index3d, int depth) nogil

cdef vector[vector[Py_ssize_t]] getIndex3dNeighbours(vector[Py_ssize_t] index3d, int depth) nogil

# Full neighbours means all 26 neighbours with diagonals
cdef vector[vector[Py_ssize_t]] getIndex3dFullNeighbours(vector[Py_ssize_t] index3d, int depth) nogil

cdef vector[vector[Py_ssize_t]] getCodeNeighbours(vector[Py_ssize_t] code) nogil

cdef vector[vector[Py_ssize_t]] getCodeFullNeighbours(vector[Py_ssize_t] code) nogil

cdef OTNode* getNode(Octree* octreePtr, vector[Py_ssize_t] code) nogil

cdef float minDistNodes(float n1_x, float n1_y, float n1_z, float n1_len, float n2_x, float n2_y, float n2_z, float n2_len) nogil

cdef int code2Int(vector[Py_ssize_t]* codePtr) nogil


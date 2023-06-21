# distutils: language=c++
#cython: language_level=3
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fmax, fmin
from libcpp.vector cimport vector
from libcpp.queue cimport queue, priority_queue
from libcpp.pair cimport pair
from libcpp.set cimport set as cset

cimport octree_defs
# import octree_defs
from octree_defs cimport OTNode, initialiseOTNodePointer, PyOTNode, Octree, initialiseOctreePointer, PyOctree, NodeDist, NodeDistQueue, NodeCost, NodeCostQueue

cimport octree_structure
from octree_structure cimport buildTree, extend_tree, code2index3d, index3d2code, getIndex3dNeighbours, getIndex3dFullNeighbours, getCodeNeighbours, \
    getCodeNeighbours, getCodeFullNeighbours, getNode, minDistNodes, code2Int

from cython.parallel import parallel, prange

# cdef extern from "<utility>" namespace "std" nogil:
#     vector[int] move(vector[int])

cdef int expand_on_boundary(Octree* octreePtr, int max_depth, float[:,:] points)

cdef void grow(Octree* octreePtr)

cdef void computeAllDists(Octree* octreePtr)

cdef float computeCost(Octree* octreePtr)

cdef float computeCostDiff(Octree* octreePtr, int nonSurfLeafIdx)

cdef float makeSingleMove(Octree* octreePtr, queue[int]* changedIdxesPtr)

cdef int makeTwoStepMove(Octree* octreePtr, int three_step)

cdef void grow_if_cost_reduce(Octree* octreePtr)



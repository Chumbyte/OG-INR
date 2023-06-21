# distutils: language=c++
#cython: language_level=3
from __future__ import print_function
import numpy as np
cimport numpy as np

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

cimport octree_structure
from octree_structure cimport serialise_tree, buildTree_from_data, buildTree, extend_tree, setNonSurfLeafs, extendNearSurfLeaves, code2index3d, index3d2code, \
    getIndex3dNeighbours, getIndex3dFullNeighbours, getCodeFullNeighbours, getNode, minDistNodes, code2Int


cimport octree_algo
from octree_algo cimport expand_on_boundary, grow, computeAllDists, computeCost, computeCostDiff, makeSingleMove, makeTwoStepMove, grow_if_cost_reduce


from cython.parallel import parallel, prange


##################################################################################################################
# Python Wrappers
##################################################################################################################

def py_createTree(int max_depth, float[:,:] points, float length, float top_pos_x, float top_pos_y, float top_pos_z, bool is2D):

    py_ot = PyOctree.new_struct() # octree struct
    py_ot._ptr.rootPtr = <OTNode*> malloc(sizeof(OTNode))
    initialiseOTNodePointer(py_ot._ptr.rootPtr)

    py_ot._ptr.surfaceLeavesPtr = new vector[OTNode*] ()
    py_ot._ptr.nonSurfaceLeavesPtr = new vector[OTNode*] ()
    py_ot._ptr.labelsPtr = new vector[int] ()
    py_ot._ptr.is2D = is2D

    print('before buildTree')
    import time
    t0 = time.time()

    num_nodes = buildTree(py_ot._ptr, py_ot._ptr.rootPtr, max_depth, 0, 0, points, length, top_pos_x, 
            top_pos_y, top_pos_z, NULL, is2D)
    print('before setNonSurfLeafs')
    setNonSurfLeafs(py_ot._ptr)
    # grow(py_ot._ptr)
    print('before extendNearSurfLeaves')
    extendNearSurfLeaves(py_ot._ptr, py_ot._ptr.rootPtr, max_depth, points)
    print('before setNonSurfLeafs')
    setNonSurfLeafs(py_ot._ptr)
    print("built octree has", num_nodes, "nodes")
    print('Finished Building ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    return py_ot

def py_extendNearSurf(PyOctree py_ot, int max_depth, float[:,:] points):
    print('before extendNearSurfLeaves')
    extendNearSurfLeaves(py_ot._ptr, py_ot._ptr.rootPtr, max_depth, points)
    print('before setNonSurfLeafs')
    setNonSurfLeafs(py_ot._ptr)

def py_extend_tree(PyOctree py_ot, int depth, float[:,:] points):
    extend_tree(py_ot._ptr, depth, points)
    print('before setNonSurfLeafs')
    setNonSurfLeafs(py_ot._ptr)
    print('before extendNearSurfLeaves')
    extendNearSurfLeaves(py_ot._ptr, py_ot._ptr.rootPtr, depth, points)
    print('before setNonSurfLeafs')
    setNonSurfLeafs(py_ot._ptr)

def py_grow(PyOctree py_ot):
    grow(py_ot._ptr)

def py_grow_and_expand(PyOctree py_ot, int max_depth, float[:,:] points):
    num_changed = 1
    while num_changed != 0:
        num_changed = expand_on_boundary(py_ot._ptr, max_depth, points)
        setNonSurfLeafs(py_ot._ptr)
        grow(py_ot._ptr)

def py_expand(PyOctree py_ot, int max_depth, float[:,:] points):
    num_changed = expand_on_boundary(py_ot._ptr, max_depth, points)

def py_computeAllDists(PyOctree py_ot):
    computeAllDists(py_ot._ptr)

def py_computeCost(PyOctree py_ot):
    return computeCost(py_ot._ptr)

def py_makeSingleMove(PyOctree py_ot):
    makeSingleMove(py_ot._ptr, NULL)

def py_makeTwoStepMove(PyOctree py_ot, int three_step=0):
    return makeTwoStepMove(py_ot._ptr, three_step)

def py_grow_if_cost_reduce(PyOctree py_ot):
    grow_if_cost_reduce(py_ot._ptr)

def py_serialise_tree(PyOctree py_ot):
    dataPtr = new vector[int] ()
    serialise_tree(py_ot._ptr, py_ot._ptr.rootPtr, dataPtr)
    return dataPtr[0]

def py_deserialise_tree(vector[int] data, float length, float top_pos_x, float top_pos_y, float top_pos_z, bool is2D):

    py_ot = PyOctree.new_struct() # octree struct
    py_ot._ptr.rootPtr = <OTNode*> malloc(sizeof(OTNode))
    initialiseOTNodePointer(py_ot._ptr.rootPtr)

    py_ot._ptr.surfaceLeavesPtr = new vector[OTNode*] ()
    py_ot._ptr.nonSurfaceLeavesPtr = new vector[OTNode*] ()
    py_ot._ptr.labelsPtr = new vector[int] ()
    py_ot._ptr.is2D = is2D

    buildTree_from_data(py_ot._ptr, &data, 0, py_ot._ptr.rootPtr, 0, 0, length, top_pos_x, top_pos_y, top_pos_z, NULL, is2D)

    return py_ot
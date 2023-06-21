# distutils: language=c++
#cython: language_level=3
from __future__ import print_function
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
# from libc.math cimport sqrt, fmax, fmin #, sqrtf, fmaxf, fminf
from libcpp.vector cimport vector
from libcpp.queue cimport queue, priority_queue
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp cimport bool

cdef extern from "<math.h>" nogil:
    float sqrtf(float)
    float fminf(float, float)
    float fmaxf(float, float)


cimport octree_defs
# import octree_defs
from octree_defs cimport OTNode, initialiseOTNodePointer, PyOTNode, Octree, initialiseOctreePointer, PyOctree, NodeDist, NodeDistQueue

from cython.parallel import parallel, prange

cdef void serialise_tree(Octree* octreePtr, OTNode* nodePtr, vector[int]* dataPtr):
    cdef Py_ssize_t i
    if nodePtr is NULL:
        dataPtr.push_back(-1) # null
    elif nodePtr.isLeaf:
        if nodePtr.isNonSurfLeaf:
            if octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] == 0:
                dataPtr.push_back(2) # outside
            else:
                dataPtr.push_back(3) # inside
        else:
            dataPtr.push_back(1) # surf leaf
    else:
        dataPtr.push_back(0) # non leaf
        for i in range(8):
            childPtr = nodePtr.childrenPtrArr[i]
            serialise_tree(octreePtr, childPtr, dataPtr)

cdef int buildTree_from_data(Octree* octreePtr, vector[int]* dataPtr, int dataIdx, OTNode* nodePtr, int depth, Py_ssize_t childIndex, 
        float length, float top_pos_x, float top_pos_y, float top_pos_z, OTNode* parentPtr, bool is2D):
    # print('in buildTree_from_data', dataIdx)
    cdef int num_nodes = 1 # building a node right now
    nodePtr.depth = depth; nodePtr.length = length
    nodePtr.top_pos_x = top_pos_x; nodePtr.top_pos_y = top_pos_y; nodePtr.top_pos_z = top_pos_z
    nodePtr.parentPtr = parentPtr
    nodePtr.isLeaf = 1  # This is a leaf, has no children (yet)
    nodePtr._tempLabel = 1
    nodePtr.is2D = is2D
    cdef float new_min_x = top_pos_x, new_min_y = top_pos_y, new_min_z = top_pos_z
    cdef float new_max_x = top_pos_x + length, new_max_y = top_pos_y + length, new_max_z = top_pos_z + length

    cdef Py_ssize_t i, j
    cdef int num
    nodePtr.codePtr = new vector[Py_ssize_t] ()
    if nodePtr.parentPtr is NULL:
        # Make Code
        nodePtr.codePtr.push_back(0)
    else:
        # Make Code
        parentCode = nodePtr.parentPtr.codePtr[0]
        nodePtr.codePtr.reserve(parentCode.size()+1)
        for i in range(parentCode.size()):
            nodePtr.codePtr.push_back(parentCode[i])
        nodePtr.codePtr.push_back(childIndex)
    
    if dataPtr[0][dataIdx] == 0:
        nodePtr.isLeaf = 0
        
        dataIdx = extend_chilren_from_data(octreePtr, nodePtr, dataPtr, dataIdx+1)
        pass
    else:
        if dataPtr[0][dataIdx] == 1:
            # Surface Leaf!
            octreePtr.surfaceLeavesPtr.push_back(nodePtr)
            nodePtr.isLeaf = 1
        else:
            # Non-surface Leaf!
            assert dataPtr[0][dataIdx] == 2 or dataPtr[0][dataIdx] == 3, dataPtr[0][dataIdx]
            octreePtr.nonSurfaceLeavesPtr.push_back(nodePtr)
            nodePtr._tempLabel = 0 if dataPtr[0][dataIdx] == 2 else 1
            octreePtr.labelsPtr.push_back(nodePtr._tempLabel)
            nodePtr.isNonSurfLeaf = 1
            nodePtr.isLeaf = 1
            nodePtr.nonSurfIndex = octreePtr.labelsPtr.size() - 1
            nodePtr.index3dPtr = new vector[Py_ssize_t] (code2index3d(nodePtr.codePtr[0]))
    # print('end buildTree_from_data', dataIdx)
    return dataIdx # last index used within scope of this function

cdef int extend_chilren_from_data(Octree* octreePtr, OTNode* parentPtr, vector[int]* dataPtr, int dataIdx):
    # print('in extend_chilren_from_data', dataIdx)
    cdef int num_nodes
    cdef float new_length
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t num
    cdef float new_top_pos_x, new_top_pos_y, new_top_pos_z
    new_length = parentPtr.length/2
    num_nodes = 0
    parentPtr.isLeaf = 0 # No longer a leaf, has children

    for i in range(2):
        for j in range(2):
            for k in range(2):
                if dataPtr[0][dataIdx] == -1:
                    dataIdx += 1
                    continue
                num = 4*i + 2*j + k
                new_top_pos_x = parentPtr.top_pos_x + i*new_length
                new_top_pos_y = parentPtr.top_pos_y + j*new_length
                new_top_pos_z = parentPtr.top_pos_z + k*new_length
                
                parentPtr.childrenPtrArr[num] = <OTNode*> malloc(sizeof(OTNode))
                initialiseOTNodePointer(parentPtr.childrenPtrArr[num])
                dataIdx = buildTree_from_data(octreePtr, dataPtr, dataIdx, parentPtr.childrenPtrArr[num], parentPtr.depth+1, num,
                            new_length, new_top_pos_x, new_top_pos_y, new_top_pos_z, parentPtr, parentPtr.is2D)
                if num != 7:
                    dataIdx +=1
    # print('end extend_chilren_from_data', dataIdx)
    return dataIdx # last index used within scope of this function


cdef int extend_tree(Octree* octreePtr, int depth, float[:,:] points):
    # delete surfaceNodes list
    cdef Py_ssize_t i
    octreePtr.surfaceLeavesPtr.clear()
    print("Size is now", octreePtr.surfaceLeavesPtr.size())
    extend_chilren(octreePtr, octreePtr.rootPtr, depth, points)
    print("Size is now", octreePtr.surfaceLeavesPtr.size())

cdef void setNonSurfLeafs(Octree* octreePtr):
    # first save labels
    cdef Py_ssize_t i
    for i in range(octreePtr.nonSurfaceLeavesPtr[0].size()):
        nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
        nodePtr._tempLabel = octreePtr.labelsPtr[0][nodePtr.nonSurfIndex]
    # delete nonSurfaceNodes list and labels list
    octreePtr.nonSurfaceLeavesPtr.clear()
    octreePtr.labelsPtr.clear()
    setNonSurfacePtrs(octreePtr, octreePtr.rootPtr, 0)

cdef void setNonSurfacePtrs(Octree* octreePtr, OTNode* nodePtr, int depth):
    cdef Py_ssize_t i
    if not nodePtr.isLeaf:
        for i in range(8):
            childPtr = nodePtr.childrenPtrArr[i]
            if childPtr is not NULL:
                setNonSurfacePtrs(octreePtr, childPtr, depth+1)
    else:
        # is a leaf
        if nodePtr.isNonSurfLeaf:
            octreePtr.nonSurfaceLeavesPtr.push_back(nodePtr)
            octreePtr.labelsPtr.push_back(nodePtr._tempLabel)
            nodePtr.nonSurfIndex = octreePtr.labelsPtr.size() - 1
            nodePtr.relevantSurfIdxesPtr = new vector[int] ()
            nodePtr.index3dPtr = new vector[Py_ssize_t] (code2index3d(nodePtr.codePtr[0]))
            if nodePtr.index3dPtr[0][0] == 0 or nodePtr.index3dPtr[0][0] == 2**depth-1 or \
                nodePtr.index3dPtr[0][1] == 0 or nodePtr.index3dPtr[0][1] == 2**depth-1 or \
                nodePtr.index3dPtr[0][2] == 0 or nodePtr.index3dPtr[0][2] == 2**depth-1:
                # Boundary of domain, set to outside
                octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0
                nodePtr._tempLabel = 0
                # print(nodePtr.index3dPtr[0], nodePtr.codePtr[0])

cdef void extendNearSurfLeaves(Octree* octreePtr, OTNode* nodePtr, int max_depth, float[:,:] points):
    cdef Py_ssize_t i, j
    cdef int num_changed = 0

    for i in range(octreePtr.surfaceLeavesPtr.size()):
        nodePtr = octreePtr.surfaceLeavesPtr[0][i]
        neighbours = getCodeFullNeighbours(nodePtr.codePtr[0])
        for j in range(neighbours.size()):
            num_changed += makeNodesToCode(octreePtr, neighbours[j], max_depth, points)
    print("extended near surface, num changed: ", num_changed)

cdef int makeNodesToCode(Octree* octreePtr, vector[Py_ssize_t] code, int max_depth, float[:,:] points):
    cdef int num_changed = 0
    cdef Py_ssize_t i, j
    cdef OTNode* nodePtr = octreePtr.rootPtr
    for i in range(1, code.size()):
        if nodePtr.isLeaf:
            label = octreePtr.labelsPtr[0][nodePtr.nonSurfIndex]
            nodePtr.isNonSurfLeaf = 0
            num_changed += extend_chilren(octreePtr, nodePtr, max_depth, points) # only does 1 depth at a time as no points inside node
            for j in range(8):
                childPtr = nodePtr.childrenPtrArr[j]
                if childPtr is not NULL:
                    childPtr._tempLabel = label
                    octreePtr.labelsPtr[0][childPtr.nonSurfIndex] = label
            nodePtr.isLeaf = 0
        if nodePtr.childrenPtrArr[code[i]] is NULL:
            return num_changed
        else:
            nodePtr = nodePtr.childrenPtrArr[code[i]]
    return num_changed



cdef int extend_chilren(Octree* octreePtr, OTNode* parentPtr, int max_depth, float[:,:] points):
    cdef int num_nodes
    cdef float new_length
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t num
    cdef float new_top_pos_x, new_top_pos_y, new_top_pos_z
    if parentPtr.isLeaf:
        if (not parentPtr.isNonSurfLeaf) and parentPtr.depth < max_depth: # this check is only needed for when extending a built tree
            # Parent is a leaf that contains points and is not at max_depth, so expand
            new_length = parentPtr.length/2
            num_nodes = 0
            parentPtr.isLeaf = 0 # No longer a leaf, has children
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        num = 4*i + 2*j + k
                        new_top_pos_x = parentPtr.top_pos_x + i*new_length
                        new_top_pos_y = parentPtr.top_pos_y + j*new_length
                        new_top_pos_z = parentPtr.top_pos_z + k*new_length
                        cond = (new_top_pos_z == 0.0) if octreePtr.is2D else True
                        if cond:
                            parentPtr.childrenPtrArr[num] = <OTNode*> malloc(sizeof(OTNode))
                            initialiseOTNodePointer(parentPtr.childrenPtrArr[num])
                            num_nodes += buildTree(octreePtr, parentPtr.childrenPtrArr[num], max_depth, parentPtr.depth+1, num, points,
                                new_length, new_top_pos_x, new_top_pos_y, new_top_pos_z, parentPtr, parentPtr.is2D)
    else:
        # Not Leaf, apply to each child
        for i in range(8):
            if parentPtr.childrenPtrArr[i] is not NULL:
                extend_chilren(octreePtr, parentPtr.childrenPtrArr[i], max_depth, points)
    return num_nodes

cdef inline int checkPointInBounds(float px, float py, float pz, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z) nogil:
    if min_x <= px and px <= max_x and min_y <= py and py <= max_y and min_z <= pz and pz <= max_z:
        return 1
    else:
        return 0

cdef int buildTree(Octree* octreePtr, OTNode* nodePtr, int max_depth, int depth, Py_ssize_t childIndex, 
        float[:,:] points, float length, 
        float top_pos_x, float top_pos_y, float top_pos_z, OTNode* parentPtr, bool is2D):
    cdef int num_nodes = 1 # building a node right now
    nodePtr.depth = depth; nodePtr.length = length
    nodePtr.top_pos_x = top_pos_x; nodePtr.top_pos_y = top_pos_y; nodePtr.top_pos_z = top_pos_z
    nodePtr.parentPtr = parentPtr
    nodePtr.isLeaf = 1  # This is a leaf, has no children (yet)
    nodePtr._tempLabel = 1
    nodePtr.is2D = is2D
    cdef float new_min_x = top_pos_x, new_min_y = top_pos_y, new_min_z = top_pos_z
    cdef float new_max_x = top_pos_x + length, new_max_y = top_pos_y + length, new_max_z = top_pos_z + length

    cdef Py_ssize_t x_max = points.shape[0], y_max = points.shape[1]
    cdef Py_ssize_t i, j

    cdef int num
    cdef Py_ssize_t num_points
    nodePtr.pointIdxesPtr = new vector[Py_ssize_t] ()
    nodePtr.codePtr = new vector[Py_ssize_t] ()
    if nodePtr.parentPtr is NULL:
        # Make Code
        nodePtr.codePtr.push_back(0)
        # Make PointIdxes
        num = 0
        num_points = x_max
        nodePtr.pointIdxesPtr.reserve(num_points)
        for i in range(num_points):
            if checkPointInBounds(points[i,0], points[i,1], points[i,2], new_min_x, new_max_x, new_min_y, new_max_y, new_min_z, new_max_z):
                num += 1
                nodePtr.pointIdxesPtr.push_back(i)
        # print('OTNode at depth', depth, 'has no parent and ', num ,'/',num_points,'points' )
    else:
        # Make Code
        parentCode = nodePtr.parentPtr.codePtr[0]
        nodePtr.codePtr.reserve(parentCode.size()+1)
        for i in range(parentCode.size()):
            nodePtr.codePtr.push_back(parentCode[i])
        nodePtr.codePtr.push_back(childIndex)
        # Make PointIdxes
        num = 0
        parentPointIdxes = nodePtr.parentPtr.pointIdxesPtr[0]
        num_points = parentPointIdxes.size()
        nodePtr.pointIdxesPtr.reserve(num_points // 8)
        for i in range(num_points):
            j = parentPointIdxes[i]
            if  checkPointInBounds(points[j,0], points[j,1], points[j,2], new_min_x, new_max_x, new_min_y, new_max_y, new_min_z, new_max_z):
                num += 1
                nodePtr.pointIdxesPtr.push_back(j)
        # print('OTNode at depth', depth, 'has parent and ', num ,'/',num_points,'points and code', nodePtr.codePtr[0] )
    nodePtr.pointIdxesPtr.shrink_to_fit()
    
    if num == 0:
        # Non-surface Leaf!
        octreePtr.nonSurfaceLeavesPtr.push_back(nodePtr)
        nodePtr._tempLabel = 1
        octreePtr.labelsPtr.push_back(1)
        nodePtr.isNonSurfLeaf = 1
        nodePtr.isLeaf = 1
        nodePtr.nonSurfIndex = octreePtr.labelsPtr.size() - 1
        nodePtr.relevantSurfIdxesPtr = new vector[int] ()
        nodePtr.index3dPtr = new vector[Py_ssize_t] (code2index3d(nodePtr.codePtr[0]))
        if nodePtr.index3dPtr[0][0] == 0 or nodePtr.index3dPtr[0][0] == 2**depth-1 or \
            nodePtr.index3dPtr[0][1] == 0 or nodePtr.index3dPtr[0][1] == 2**depth-1 or \
            nodePtr.index3dPtr[0][2] == 0 or nodePtr.index3dPtr[0][2] == 2**depth-1:
            # Boundary of domain, set to outside
            nodePtr._tempLabel = 0
            octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0
            # print(nodePtr.index3dPtr[0], nodePtr.codePtr[0])
        return num_nodes
    else:
        if nodePtr.depth == max_depth:
            # Surface Leaf!
            octreePtr.surfaceLeavesPtr.push_back(nodePtr)
            nodePtr.isLeaf = 1
            return num_nodes
        else:
            # Non-leaf Node!
            num_nodes += extend_chilren(octreePtr, nodePtr, max_depth, points)
            return num_nodes

cdef vector[Py_ssize_t] code2index3d(vector[Py_ssize_t] code) nogil:
    cdef vector[Py_ssize_t] index3d
    index3d.push_back(0)
    index3d.push_back(0)
    index3d.push_back(0)
    cdef Py_ssize_t i
    cdef Py_ssize_t clen = code.size()
    for i in range(clen):
        index3d[2] += ((code[clen-1-i])      & 1) * 2 ** i
        index3d[1] += ((code[clen-1-i] >> 1) & 1) * 2 ** i
        index3d[0] += ((code[clen-1-i] >> 2) & 1) * 2 ** i
    return index3d

def py_code2index3d(py_code):
    cdef vector[Py_ssize_t] code = vector[Py_ssize_t]()
    for num in py_code:
        code.push_back(num)
    index3d = code2index3d(code)
    return np.array(index3d)

cdef vector[Py_ssize_t] index3d2code(vector[Py_ssize_t] index3d, int depth) nogil:
    # depth starts from 0, so should be depth + 1
    cdef vector[Py_ssize_t] code = vector[Py_ssize_t](depth+1, 0)
    cdef Py_ssize_t i, j
    for i in range(3):
        for j in range(depth): # not depth + 1 here as the first value is always 0 anyway
            code[depth-j] += ((index3d[2-i] >> j) & 1) * 2 ** i
    # print(code, index3d)
    return code

cdef vector[vector[Py_ssize_t]] getIndex3dNeighbours(vector[Py_ssize_t] index3d, int depth) nogil:
    cdef vector[vector[Py_ssize_t]] neighbours = vector[vector[Py_ssize_t]]()
    cdef Py_ssize_t i, j, direction, offset, new_val
    cdef Py_ssize_t max_val = 2 ** depth
    cdef vector[Py_ssize_t] neigh
    for i in range(6):
        direction = i // 2 # 0,1 or 2
        offset = (i % 2) * 2 - 1 # -1 or 1
        neigh = vector[Py_ssize_t](index3d)
        new_val = neigh[direction] + offset
        if new_val < 0 or new_val >= max_val:
            continue
        neigh[direction] = new_val
        neighbours.push_back(neigh)
        # print('n', i, direction, offset, neigh)
    # print(neighbours)
    return neighbours

# Full neighbours means all 26 neighbours with diagonals
cdef vector[vector[Py_ssize_t]] getIndex3dFullNeighbours(vector[Py_ssize_t] index3d, int depth) nogil:
    cdef vector[vector[Py_ssize_t]] neighbours = vector[vector[Py_ssize_t]]()
    cdef Py_ssize_t i, j, k, val_i, val_j, val_k
    cdef Py_ssize_t max_val = 2 ** depth
    cdef vector[Py_ssize_t] neigh
    for i in range(-1,2):
        val_i = index3d[0] + i
        if val_i < 0 or val_i >= max_val:
            continue
        for j in range(-1,2):
            val_j = index3d[1] + j
            if val_j < 0 or val_j >= max_val:
                continue
            for k in range(-1,2):
                val_k = index3d[2] + k
                if val_k < 0 or val_k >= max_val:
                    continue
                if val_i == 0 and val_j == 0 and val_k == 0:
                    continue
                neigh = vector[Py_ssize_t](index3d)
                neigh[0] = val_i
                neigh[1] = val_j
                neigh[2] = val_k
                neighbours.push_back(neigh)
    # print(neighbours)
    return neighbours

cdef vector[vector[Py_ssize_t]] getCodeNeighbours(vector[Py_ssize_t] code) nogil:
    cdef Py_ssize_t depth = code.size() - 1
    indexd3d = code2index3d(code)
    index3d_neighbours = getIndex3dNeighbours(indexd3d, depth)
    cdef vector[vector[Py_ssize_t]] neighbours = vector[vector[Py_ssize_t]](index3d_neighbours.size())
    cdef Py_ssize_t i
    for i in range(index3d_neighbours.size()):
        neighbours[i] = index3d2code(index3d_neighbours[i], depth)
    # print(code, depth, indexd3d, '\n\t', index3d_neighbours, '\n\t', neighbours)
    return neighbours

cdef vector[vector[Py_ssize_t]] getCodeFullNeighbours(vector[Py_ssize_t] code) nogil:
    cdef Py_ssize_t depth = code.size() - 1
    indexd3d = code2index3d(code)
    index3d_neighbours = getIndex3dFullNeighbours(indexd3d, depth)
    cdef vector[vector[Py_ssize_t]] neighbours = vector[vector[Py_ssize_t]](index3d_neighbours.size())
    cdef Py_ssize_t i
    for i in range(index3d_neighbours.size()):
        neighbours[i] = index3d2code(index3d_neighbours[i], depth)
    # print(code, depth, indexd3d, '\n\t', index3d_neighbours, '\n\t', neighbours)
    return neighbours

cdef OTNode* getNode(Octree* octreePtr, vector[Py_ssize_t] code) nogil:
    # assert code.size() > 0
    # assert code[0] == 0
    cdef Py_ssize_t i
    cdef OTNode* nodePtr = octreePtr.rootPtr
    for i in range(1, code.size()):
        if nodePtr.isLeaf:
            break
        if nodePtr.childrenPtrArr[code[i]] is NULL:
            return NULL
            break
        nodePtr = nodePtr.childrenPtrArr[code[i]]
    return nodePtr

cdef float minDistNodes(float n1_x, float n1_y, float n1_z, float n1_len, float n2_x, float n2_y, float n2_z, float n2_len) nogil:
    cdef float u_x = fmaxf(0.0, n1_x - (n2_x + n2_len))
    cdef float u_y = fmaxf(0.0, n1_y - (n2_y + n2_len))
    cdef float u_z = fmaxf(0.0, n1_z - (n2_z + n2_len))
    cdef float v_x = fmaxf(0.0, n2_x - (n1_x + n1_len))
    cdef float v_y = fmaxf(0.0, n2_y - (n1_y + n1_len))
    cdef float v_z = fmaxf(0.0, n2_z - (n1_z + n1_len))
    return sqrtf(u_x*u_x + u_y*u_y + u_z*u_z + v_x*v_x + v_y*v_y + v_z*v_z)

cdef int code2Int(vector[Py_ssize_t]* codePtr) nogil:
    # code to unique int
    cdef Py_ssize_t i
    cdef int val = 0
    for i in range(codePtr.size()):
        val += codePtr[0][codePtr.size()-i-1] * (8**i)
    return val




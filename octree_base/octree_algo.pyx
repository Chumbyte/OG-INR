# distutils: language=c++
#cython: language_level=3
from __future__ import print_function
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free, calloc
# from libc.math cimport sqrt, fmax, fmin
from libcpp.vector cimport vector
from libcpp.queue cimport queue, priority_queue
from libcpp.pair cimport pair
from libcpp.set cimport set as cset

cdef extern from "<math.h>" nogil:
    float sqrtf(float)
    float fminf(float, float)
    float fmaxf(float, float)

cimport octree_defs
# import octree_defs
from octree_defs cimport OTNode, initialiseOTNodePointer, PyOTNode, Octree, initialiseOctreePointer, PyOctree, NodeDist, NodeDistQueue, NodeCost, NodeCostQueue

cimport octree_structure
from octree_structure cimport buildTree, extend_tree, extend_chilren, code2index3d, index3d2code, getIndex3dNeighbours, getIndex3dFullNeighbours, getCodeNeighbours, \
    getCodeNeighbours, getCodeFullNeighbours, getNode, minDistNodes, code2Int

from cython.parallel import parallel, prange

cdef extern from "<utility>" namespace "std" nogil:
    vector[int] move(vector[int])

cdef int expand_on_boundary(Octree* octreePtr, int max_depth, float[:,:] points):
    cdef Py_ssize_t i, j
    cdef int num_changed = 0

    for i in range(octreePtr.labelsPtr.size()):
        if octreePtr.labelsPtr[0][i] == 1:
            # inside
            nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
            if nodePtr.depth == max_depth:
                continue
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                    # neighbour is outside
                    # inside leaf (nodePtr) not at max_depth is on the boundary, expand children
                    label = octreePtr.labelsPtr[0][nodePtr.nonSurfIndex]
                    nodePtr.isNonSurfLeaf = 0
                    num_changed += extend_chilren(octreePtr, nodePtr, max_depth, points) # only does 1 depth at a time as no points inside node
                    for j in range(8):
                        childPtr = nodePtr.childrenPtrArr[j]
                        if childPtr is not NULL:
                            childPtr._tempLabel = label
                            octreePtr.labelsPtr[0][childPtr.nonSurfIndex] = label
                    nodePtr.isLeaf = 0
        if octreePtr.labelsPtr[0][i] == 0:
            # outside
            nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.depth == max_depth:
                    continue
                if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
                    # neighbour is inside
                    # inside leaf (neigh_nodePtr) not at max_depth is on the boundary, expand children
                    label = octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex]
                    neigh_nodePtr.isNonSurfLeaf = 0
                    num_changed += extend_chilren(octreePtr, neigh_nodePtr, max_depth, points) # only does 1 depth at a time as no points inside node
                    for j in range(8):
                        childPtr = neigh_nodePtr.childrenPtrArr[j]
                        if childPtr is not NULL:
                            childPtr._tempLabel = label
                            octreePtr.labelsPtr[0][childPtr.nonSurfIndex] = label
                    neigh_nodePtr.isLeaf = 0
    print("extended on boundary, num changed:", num_changed)
    return num_changed


cdef void grow(Octree* octreePtr):
    cdef queue[OTNode*] nodePtrQueue
    cdef Py_ssize_t i, j
    cdef int num_changed = 0
    cdef int count
    cdef queue[OTNode*] innerNodePtrQueue

    for i in range(octreePtr.labelsPtr.size()):
        if octreePtr.labelsPtr[0][i] == 1:
            # inside
            nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                    # neighbour is outside
                    nodePtrQueue.push(nodePtr)
    # print("queue len:", nodePtrQueue.size())
    while not nodePtrQueue.empty():
        # print('size', nodePtrQueue.size())
        nodePtr = nodePtrQueue.front()
        # neighbours = getCodeNeighbours(nodePtr.codePtr[0])
        neighbours = getCodeFullNeighbours(nodePtr.codePtr[0])
        if octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] == 0:
            # already changed to outside
            nodePtrQueue.pop()
            continue
        count = 0
        for i in range(neighbours.size()):
            neigh_nodePtr = getNode(octreePtr, neighbours[i])
            if neigh_nodePtr is NULL:
                continue
            if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                # inside node have outside neighbour
                count += 1
            # if count == (2 if octreePtr.is2D else 3):
            # if count == (4 if octreePtr.is2D else 8):
            if count == (5 if octreePtr.is2D else 14):
                # set to outside
                octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0
                num_changed += 1
                # now add all its inside neighbours to the queue
                # neighbours = getCodeNeighbours(neigh_node.codePtr[0])
                for j in range(neighbours.size()):
                    # neigh_nodePtr2 = getNode(octreePtr, neighbours[j])
                    # # if not neigh_nodePtr2.isLeaf:                    
                    # if neigh_nodePtr2.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr2.nonSurfIndex] == 1:
                    #     # neighbour is inside
                    #     nodePtrQueue.push(neigh_nodePtr2)
                    neigh_nodePtr = getNode(octreePtr, neighbours[j])
                    if neigh_nodePtr is NULL:
                        continue
                    innerNodePtrQueue.push(neigh_nodePtr)
                    while not innerNodePtrQueue.empty():
                        innerNodePtr = innerNodePtrQueue.front()
                        # if innerNodePtr is NULL:
                        #     innerNodePtrQueue.pop()
                        #     continue
                        if not innerNodePtr.isLeaf:
                            for childPtr in innerNodePtr.childrenPtrArr:
                                if childPtr is not NULL:
                                    innerNodePtrQueue.push(childPtr)
                        else:
                            if innerNodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 1:
                                # Neighbour (or child of neighbour which might not be a neighbour) is inside
                                nodePtrQueue.push(innerNodePtr)
                        innerNodePtrQueue.pop()
                break
        nodePtrQueue.pop()
    print(f"At the end of grow, num_changed={num_changed}")

cdef void computeAllDists(Octree* octreePtr):
    cdef NodeDist* item
    cdef NodeDist* item2
    cdef float dist
    cdef cset[int]* idxSet_local
    cdef queue[OTNode*]* neighQueue_local
    cdef NodeDistQueue* nd_queue_local
    cdef Py_ssize_t i, j
    cdef int val
    cdef vector[vector[Py_ssize_t]] neighbours
    cdef OTNode* neigh_nodePtr

    # multithreaded from here
    with nogil, parallel(num_threads=10):
        idxSet_local = new cset[int]()
        neighQueue_local = new queue[OTNode*]()
        nd_queue_local = new NodeDistQueue()

        for i in prange(octreePtr.surfaceLeavesPtr.size()):
            surfPtr = octreePtr.surfaceLeavesPtr[0][i]
            surfPtr.closestNonSurfIdxesPtr = new vector[NodeDist]()
            neighbours = getCodeFullNeighbours(surfPtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                val = code2Int(neigh_nodePtr.codePtr)
                if not idxSet_local.count(val):
                    idxSet_local.insert(val)
                    neighQueue_local.push(neigh_nodePtr)
            while (not neighQueue_local.empty()) and (nd_queue_local.size() < 100):
                nodePtr = neighQueue_local.front()
                if not nodePtr.isNonSurfLeaf:
                    if not nodePtr.isLeaf:
                        for j in range(8):
                            childPtr = nodePtr.childrenPtrArr[j]
                            if childPtr is NULL:
                                continue
                            val = code2Int(childPtr.codePtr)
                            if not idxSet_local.count(val):
                                idxSet_local.insert(val)
                                neighQueue_local.push(childPtr)
                    neighQueue_local.pop()
                    continue
                dist = minDistNodes(surfPtr.top_pos_x, surfPtr.top_pos_y, surfPtr.top_pos_z, surfPtr.length, 
                        nodePtr.top_pos_x, nodePtr.top_pos_y, nodePtr.top_pos_z, nodePtr.length)
                item = <NodeDist*> malloc(sizeof(NodeDist)) 
                item.nonSurfIndex = nodePtr.nonSurfIndex
                item.dist = dist
                nd_queue_local.push(item) # add to surf's closest nonsurf list
                neighbours = getCodeFullNeighbours(nodePtr.codePtr[0])
                for j in range(neighbours.size()):
                    neigh_nodePtr = getNode(octreePtr, neighbours[j])
                    if neigh_nodePtr is NULL:
                        continue
                    val = code2Int(neigh_nodePtr.codePtr)
                    if not idxSet_local.count(val):
                        idxSet_local.insert(val)
                        neighQueue_local.push(neigh_nodePtr)
                neighQueue_local.pop()
            # if i % 1000 == 0:
            #     print("it {:6d}/{}, queue size {}, set size {}".format(i, octreePtr.surfaceLeavesPtr.size(), neighQueue_local.size(), idxSet.size()))
            idxSet_local.clear()
            while (not neighQueue_local.empty()):
                neighQueue_local.pop()
            while (nd_queue_local.size() > 50):
                free(nd_queue_local.top())
                nd_queue_local.pop()
            with gil:
                while not nd_queue_local.empty():
                    item2 = nd_queue_local.top()
                    surfPtr.closestNonSurfIdxesPtr.push_back(item2[0])
                    nonSurfLeafPtr = octreePtr.nonSurfaceLeavesPtr[0][item2.nonSurfIndex]
                    nonSurfLeafPtr.relevantSurfIdxesPtr.push_back(i)
                    nd_queue_local.pop()
        with gil:
            free(nd_queue_local)
            free(idxSet_local)
            free(neighQueue_local)

cdef float computeCost(Octree* octreePtr):
    # First computer surface cost
    cdef float surface_cost = 0.0
    cdef float min_dist_inside
    cdef float min_dist_outside
    cdef Py_ssize_t i, j

    # for i in range(octreePtr.surfaceLeavesPtr.size()):
    # # for i in prange(octreePtr.surfaceLeavesPtr.size(), nogil=True):
    #     surfPtr = octreePtr.surfaceLeavesPtr[0][i]
    #     min_dist_inside = 0.5
    #     min_dist_outside = 0.5
    #     for j in range(surfPtr.closestNonSurfIdxesPtr.size()):
    #         # item = surfPtr.closestNonSurfIdxesPtr[0][j]
    #         if octreePtr.labelsPtr[0][surfPtr.closestNonSurfIdxesPtr[0][j].nonSurfIndex]:
    #             min_dist_inside = fminf(min_dist_inside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #         else:
    #             min_dist_outside = fminf(min_dist_outside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #     # surface_cost += min_dist_inside + min_dist_outside
    #     # surface_cost += min_dist_outside
    #     surface_cost += min_dist_outside**2
    #     # surface_cost += min_dist_inside**2 + min_dist_outside**2
    #     # print(min_dist_inside, min_dist_outside)
    # # print("Surface Cost:", surface_cost)

    cdef float surf_cost = 0.0
    cdef float inside_surf_cost = 0.0
    cdef float outside_surf_cost = 0.0
    cdef int count, count_outside
    cdef int inside_thresh = (4 if octreePtr.is2D else 11)
    cdef int outside_thresh = (3 if octreePtr.is2D else 9)
    for i in range(octreePtr.surfaceLeavesPtr.size()):
    # for i in prange(octreePtr.surfaceLeavesPtr.size(), nogil=True):
        surfPtr = octreePtr.surfaceLeavesPtr[0][i]
        neighbours = getCodeFullNeighbours(surfPtr.codePtr[0])
        count = 0
        count_outside = 0
        for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isLeaf and not neigh_nodePtr.isNonSurfLeaf:
                    count += 1 # neighbour is also a surface leaf
                    count_outside += 1
                if neigh_nodePtr.isNonSurfLeaf:
                    if octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
                        count += 1 # neighbour is inside
                    else:
                        count_outside += 4 # neighbour is outside
        if count < inside_thresh:
            # inside_surf_cost += (inside_thresh - count)*10*surfPtr.length*100
            inside_surf_cost += (inside_thresh*(1 - count/inside_thresh)**1)*10*surfPtr.length*100
        if count_outside < outside_thresh:
            # outside_surf_cost += (outside_thresh - count_outside)*10*surfPtr.length*100
            outside_surf_cost += (outside_thresh*(1 - count_outside/outside_thresh)**1)*10*surfPtr.length*100
            # surf_cost += surfPtr.length*100
            # print('hi', count)
    # print("Surface Cost2:", surf_cost)
    surface_cost += inside_surf_cost + outside_surf_cost
    # print("cost", count, surf_cost)
                    

    cdef float border_cost = 0

    for i in range(octreePtr.nonSurfaceLeavesPtr.size()):
        nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
        node_label = octreePtr.labelsPtr[0][i]
        # if octreePtr.labelsPtr[0][i] == 1:
        #     # inside
            
        neighbours = getCodeNeighbours(nodePtr.codePtr[0])
        for j in range(neighbours.size()):
            neigh_nodePtr = getNode(octreePtr, neighbours[j])
            if neigh_nodePtr is NULL:
                continue
            neigh_node_label = octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex]
            if neigh_nodePtr.isNonSurfLeaf and node_label != neigh_node_label:
                length1 = nodePtr.length
                length2 = neigh_nodePtr.length
                if length1 == length2:
                    border_cost += length1 / 2 # as there will be a component from the other side
                    # border_cost += length1**2 / 2 # as there will be a component from the other side
                else:
                    border_cost += fminf(length1, length2)
                    # border_cost += fminf(length1, length2)**2
    # print("Border Cost:", border_cost)
    # print("Cost:", surface_cost + border_cost, surface_cost + 25*border_cost)
    if octreePtr.is2D:
        return surface_cost + 5*1*border_cost
    else:
        print(f'\t Cost breakdown: Surf {surface_cost} ({inside_surf_cost},{outside_surf_cost}), border {5*1*1*border_cost}')
        return surface_cost + 5*1*1*border_cost
    # TRIED 0.8, 1, 2, 5 and 25. 0.8/1 were sometimes better...

cdef float computeCostDiff(Octree* octreePtr, int nonSurfLeafIdx):
    # First computer surface cost
    cdef float initial_surface_cost = 0.0
    cdef float after_surface_cost = 0.0
    cdef float initial_min_dist_inside, initial_min_dist_outside, after_min_dist_inside, after_min_dist_outside
    cdef Py_ssize_t i, j
    nonSurfLeafPtr = octreePtr.nonSurfaceLeavesPtr[0][nonSurfLeafIdx]

    # for i in range(nonSurfLeafPtr.relevantSurfIdxesPtr.size()): # for each surf leaf relevant to the current nonSurfLeaf
    #     surfPtr = octreePtr.surfaceLeavesPtr[0][nonSurfLeafPtr.relevantSurfIdxesPtr[0][i]]
    #     initial_min_dist_inside = 0.5
    #     initial_min_dist_outside = 0.5
    #     after_min_dist_inside = 0.5
    #     after_min_dist_outside = 0.5
    #     for j in range(surfPtr.closestNonSurfIdxesPtr.size()): # for every non surf leaf close to that surf leaf
    #         if surfPtr.closestNonSurfIdxesPtr[0][j].nonSurfIndex == nonSurfLeafIdx: # if it is the current nonSurfLeaf
    #             # consider inside for initial, outside for after
    #             initial_min_dist_inside = fminf(initial_min_dist_inside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #             after_min_dist_outside = fminf(after_min_dist_outside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #         else: # do as normal
    #             if octreePtr.labelsPtr[0][surfPtr.closestNonSurfIdxesPtr[0][j].nonSurfIndex]: # If inside
    #                 initial_min_dist_inside = fminf(initial_min_dist_inside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #                 after_min_dist_inside = fminf(after_min_dist_inside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #             else:
    #                 initial_min_dist_outside = fminf(initial_min_dist_outside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #                 after_min_dist_outside = fminf(after_min_dist_outside, surfPtr.closestNonSurfIdxesPtr[0][j].dist)
    #     # initial_surface_cost += initial_min_dist_inside + initial_min_dist_outside
    #     # after_surface_cost += after_min_dist_inside + after_min_dist_outside
    #     # initial_surface_cost += initial_min_dist_outside
    #     # after_surface_cost += after_min_dist_outside
    #     initial_surface_cost += initial_min_dist_outside**2
    #     after_surface_cost += after_min_dist_outside**2
    #     # initial_surface_cost += initial_min_dist_inside**2 + initial_min_dist_outside**2
    #     # after_surface_cost += after_min_dist_inside**2 + after_min_dist_outside**2
    

    cdef float initial_surf_cost = 0.0
    cdef float after_surf_cost = 0.0
    cdef int count_before, count_after, count_outside_before, count_outside_after
    cdef int inside_thresh = (4 if octreePtr.is2D else 11)
    cdef int outside_thresh = (3 if octreePtr.is2D else 9)
    for i in range(nonSurfLeafPtr.relevantSurfIdxesPtr.size()): # for each surf leaf relevant to the current nonSurfLeaf
        surfPtr = octreePtr.surfaceLeavesPtr[0][nonSurfLeafPtr.relevantSurfIdxesPtr[0][i]]
        neighbours = getCodeFullNeighbours(surfPtr.codePtr[0])
        count_before = 0
        count_after = 0
        count_outside_before = 0
        count_outside_after = 0
        for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isLeaf and not neigh_nodePtr.isNonSurfLeaf:
                    count_before += 1 # neighbour is also a surface leaf
                    count_after += 1 # neighbour is also a surface leaf
                    count_outside_before += 1 # neighbour is outside
                    count_outside_after += 1 # neighbour is outside
                if neigh_nodePtr.isNonSurfLeaf:
                    if neigh_nodePtr.nonSurfIndex == nonSurfLeafIdx: # if it is the current nonSurfLeaf
                        # consider inside for initial, outside for after
                        count_before += 1 # neighbour is inside initially
                        count_outside_after += 4 # neighbour is possibly chnaged to outside
                    else:
                        if octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
                            count_before += 1 # neighbour is inside
                            count_after += 1 # neighbour is inside
                        else:
                            count_outside_before += 4 # neighbour is outside
                            count_outside_after += 4 # neighbour is outside
        if count_before < inside_thresh: # as neighbours include self
            initial_surf_cost += (inside_thresh - count_before)*10*surfPtr.length*100
            # initial_surf_cost += (inside_thresh*(1 - count_before/inside_thresh)**1)*10*surfPtr.length*100
        if count_after < inside_thresh: # as neighbours include self
            after_surf_cost += (inside_thresh - count_after)*10*surfPtr.length*100
            # after_surf_cost += (inside_thresh*(1 - count_after/inside_thresh)**1)*10*surfPtr.length*100
        if count_outside_before < outside_thresh: # as neighbours include self
            initial_surf_cost += (outside_thresh - count_outside_before)*10*surfPtr.length*100
            # initial_surf_cost += (outside_thresh*(1 - count_outside_before/outside_thresh)**1)*10*surfPtr.length*100
        if count_outside_after < outside_thresh: # as neighbours include self
            after_surf_cost += (outside_thresh - count_outside_after)*10*surfPtr.length*100
            # after_surf_cost += (outside_thresh*(1 - count_outside_after/outside_thresh)**1)*10*surfPtr.length*100
    # print("Surface Cost2:", surf_cost)
    initial_surface_cost += initial_surf_cost
    after_surface_cost += after_surf_cost

    cdef float initial_border_cost = 0
    cdef float after_border_cost = 0
    cdef queue[OTNode*] innerNodePtrQueue
    cdef float dist

    neighbours = getCodeNeighbours(nonSurfLeafPtr.codePtr[0])
    for i in range(neighbours.size()): # for each neigh of current nonSurfLeaf
        neigh_nodePtr = getNode(octreePtr, neighbours[i])
        if neigh_nodePtr is NULL:
            continue
        if neigh_nodePtr.isLeaf: # neigh in that direction is same depth or lower
            if neigh_nodePtr.isNonSurfLeaf:
                length1 = nonSurfLeafPtr.length
                length2 = neigh_nodePtr.length
                if octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex]: # inside, so cost with after case
                    after_border_cost += fminf(nonSurfLeafPtr.length, neigh_nodePtr.length) # no need for case when length equal as only looking from one side
                    # after_border_cost += fminf(nonSurfLeafPtr.length, neigh_nodePtr.length)**2 # no need for case when length equal as only looking from one side
                else: # outside, so cost with initial case
                    initial_border_cost += fminf(nonSurfLeafPtr.length, neigh_nodePtr.length) # no need for case when length equal as only looking from one side
                    # initial_border_cost += fminf(nonSurfLeafPtr.length, neigh_nodePtr.length)**2 # no need for case when length equal as only looking from one side
        else:
            # consider all neighbours in that direction, could be many children. Do this in a queue
            innerNodePtrQueue.push(neigh_nodePtr)
            while not innerNodePtrQueue.empty():
                innerNodePtr = innerNodePtrQueue.front()
                if not innerNodePtr.isLeaf:
                    for childPtr in innerNodePtr.childrenPtrArr:
                        if childPtr is not NULL:
                            innerNodePtrQueue.push(childPtr)
                else:
                    if innerNodePtr.isNonSurfLeaf: 
                        # now need to make sure it borders current nonSurfLeaf
                        dist = minDistNodes(nonSurfLeafPtr.top_pos_x, nonSurfLeafPtr.top_pos_y, nonSurfLeafPtr.top_pos_z, nonSurfLeafPtr.length, 
                                innerNodePtr.top_pos_x, innerNodePtr.top_pos_y, innerNodePtr.top_pos_z, innerNodePtr.length)
                        if dist < 1e-6: # dist is essentially 0 so they are touching
                            length1 = nonSurfLeafPtr.length
                            length2 = innerNodePtr.length
                            if octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex]: # inside, so cost with after case
                                after_border_cost += fminf(nonSurfLeafPtr.length, innerNodePtr.length) # no need for case when length equal as only looking from one side
                                # after_border_cost += fminf(nonSurfLeafPtr.length, innerNodePtr.length)**2 # no need for case when length equal as only looking from one side
                            else: # outside, so cost with initial case
                                initial_border_cost += fminf(nonSurfLeafPtr.length, innerNodePtr.length) # no need for case when length equal as only looking from one side
                                # initial_border_cost += fminf(nonSurfLeafPtr.length, innerNodePtr.length)**2 # no need for case when length equal as only looking from one side

                innerNodePtrQueue.pop()
    
    # print("Border Cost:", border_cost)
    # print("Cost:", surface_cost + border_cost, surface_cost + 25*border_cost)
    if octreePtr.is2D:
        initial_cost = initial_surface_cost + 5*1*initial_border_cost
        after_cost = after_surface_cost + 5*1*after_border_cost
    else:
        initial_cost = initial_surface_cost + 5*1*1*initial_border_cost
        after_cost = after_surface_cost + 5*1*1*after_border_cost

    return after_cost - initial_cost # make move if this is <= 0


cdef float makeSingleMove(Octree* octreePtr, queue[int]* changedIdxesPtr):
    # cdef float currentCost = computeCost(octreePtr)
    cdef float costChange = 0
    cdef float neg_cost_diff
    cdef queue[OTNode*] nodePtrQueue
    cdef Py_ssize_t i, j
    cdef int num_changed = 0
    cdef int* inQueue = <int*> malloc(octreePtr.labelsPtr.size() * sizeof(int))
    cdef queue[OTNode*] innerNodePtrQueue
    # initial_cost = computeCost(octreePtr)
    # print('1step: initial_cost', initial_cost)

    if changedIdxesPtr is NULL:
        for i in range(octreePtr.labelsPtr.size()):
            inQueue[i] = 0
            if octreePtr.labelsPtr[0][i] == 1:
                # inside
                nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
                neighbours = getCodeNeighbours(nodePtr.codePtr[0])
                for j in range(neighbours.size()):
                    neigh_nodePtr = getNode(octreePtr, neighbours[j])
                    if neigh_nodePtr is NULL:
                        continue
                    if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                        # neighbour is outside
                        if not inQueue[neigh_nodePtr.nonSurfIndex]:
                            nodePtrQueue.push(nodePtr)
                            inQueue[i] = 1
                            break
    else:
        while not changedIdxesPtr.empty():
            nodePtr = octreePtr.nonSurfaceLeavesPtr[0][changedIdxesPtr.front()]
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                    # neighbour is outside
                    if not inQueue[neigh_nodePtr.nonSurfIndex]:
                        nodePtrQueue.push(nodePtr)
                        inQueue[changedIdxesPtr.front()] = 1
                        break
            changedIdxesPtr.pop()
    # print("queue len:", nodePtrQueue.size())
    while not nodePtrQueue.empty():
        # print('size', nodePtrQueue.size())
        # currentCost = computeCost(octreePtr)
        nodePtr = nodePtrQueue.front()
        nodeIdx = nodePtr.nonSurfIndex
        if octreePtr.labelsPtr[0][nodeIdx] == 0:
            # has already been changed in a prev iter by its neighbour
            nodePtrQueue.pop()
            continue

        # Trying making a move
        # print(f"Extracted node {nodeIdx}")
        # assert octreePtr.labelsPtr[0][nodeIdx] == 1
        # inner_initial_cost = computeCost(octreePtr)
        # print('1step: inner_initial_cost', inner_initial_cost)
        inQueue[nodeIdx] = 0
        octreePtr.labelsPtr[0][nodeIdx] = 0 # try changing to outside
        # new_cost = computeCost(octreePtr)
        neg_cost_diff = computeCostDiff(octreePtr, nodeIdx)
        # after_cost = computeCost(octreePtr)
        # actual_cc = after_cost-inner_initial_cost
        # if abs(neg_cost_diff-actual_cc) > 1e-3:
        #     print('1step neg_cost_diff', neg_cost_diff, actual_cc, inner_initial_cost, after_cost, nodeIdx)
        # if new_cost > currentCost:
        if neg_cost_diff >= 0:
            # reject move and change back
            octreePtr.labelsPtr[0][nodeIdx] = 1
            # new_cost = computeCost(octreePtr)
            # ddd = new_cost-currentCost
            # if abs(ddd) > 1e-4:
            #     print('rejected, before cost = {:.5f}, after_cost = {:.5f}, diff = {:.5f}, pred diff = {:.5f}'.format(currentCost, new_cost, new_cost-currentCost, 0))
            #     print('diff_of_diffs = {:.5f}'.format(new_cost-currentCost))
        else:
            octreePtr.labelsPtr[0][nodeIdx] = 0
            # new_cost = computeCost(octreePtr)
            # ddd = new_cost-currentCost - neg_cost_diff
            # if abs(ddd) > 1e-4:
            #     print('accepted, before cost = {:.5f}, after_cost = {:.5f}, diff = {:.5f}, pred diff = {:.5f}'.format(currentCost, new_cost, new_cost-currentCost, neg_cost_diff))
            #     print('diff_of_diffs = {:.5f}'.format(new_cost-currentCost - neg_cost_diff))
            # print("accepted", currentCost, new_cost, nodePtrQueue.size())
            # print("accepted", currentCost, nodePtrQueue.size())
            # # accept move and update current cost
            # currentCost = new_cost
            costChange += neg_cost_diff
            num_changed += 1
            # also add new neighbours
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                innerNodePtrQueue.push(neigh_nodePtr)
                while not innerNodePtrQueue.empty():
                    innerNodePtr = innerNodePtrQueue.front()
                    if not innerNodePtr.isLeaf:
                        for childPtr in innerNodePtr.childrenPtrArr:
                            if childPtr is not NULL:
                                innerNodePtrQueue.push(childPtr)
                    else:
                        if innerNodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 1:
                            # neighbour is inside
                            if not inQueue[innerNodePtr.nonSurfIndex]:
                                nodePtrQueue.push(innerNodePtr)
                                inQueue[innerNodePtr.nonSurfIndex] = 1
                                # print(f'added node {innerNodePtr.nonSurfIndex}')
                    innerNodePtrQueue.pop()
        nodePtrQueue.pop()
    # print(f"At the end of single move making, num_changed={num_changed}")
    # print("New cost = {:.5f}".format(currentCost))
    free(inQueue)
    # final_cost = computeCost(octreePtr)
    # # print('1step: final_cost', final_cost)
    # actual_cc = final_cost - initial_cost
    # if abs(actual_cc - costChange) > 1e-4:
    #     print('###### single', costChange, actual_cc)
    return costChange


cdef int makeTwoStepMove(Octree* octreePtr, int three_step):
    cdef float currentCost
    cdef queue[OTNode*] nodePtrQueue
    cdef Py_ssize_t i, j, k
    cdef int num_changed = 0
    cdef int* inQueue = <int*> malloc(octreePtr.labelsPtr.size() * sizeof(int))
    # cdef queue[int]* changedIdxesPtr = new queue[int] ()
    cdef vector[int]* oldLabelsPtr
    cdef float costChange = 0
    cdef float neg_cost_diff
    cdef int sumVal
    # cdef queue[OTNode*] innerNodePtrQueue


    cdef vector[int]* changedIdxesPtr = new vector[int] ()
    cdef int* innerInQueue = <int*> calloc(octreePtr.labelsPtr.size(), sizeof(int)) # initialises to 0
    cdef float min_cost
    cdef int best_num_moves
    cdef NodeCost* item
    cdef NodeCostQueue innerNodeCostPtrQueue

    # cdef float initial_cost = computeCost(octreePtr)
    # print('2step: initial_cost', initial_cost)

    for i in range(octreePtr.labelsPtr.size()):
        inQueue[i] = 0
        if octreePtr.labelsPtr[0][i] == 1:
            # inside
            nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                    # neighbour is outside
                    if not inQueue[neigh_nodePtr.nonSurfIndex]:
                        nodePtrQueue.push(nodePtr)
                        inQueue[i] = 1
                        break
    # print("queue len:", nodePtrQueue.size())
    while not nodePtrQueue.empty():

        # # sumVal = 0
        # # for i in range(octreePtr.labelsPtr.size()):
        # #     sumVal += octreePtr.labelsPtr[0][i]
        # # # print(sumVal, 'before move')
        # # currentCost = computeCost(octreePtr)

        # # print('size', nodePtrQueue.size())
        # nodePtr = nodePtrQueue.front()
        # oldLabelsPtr = new vector[int] (octreePtr.labelsPtr[0])
        # # Trying making a move
        # nodeIdx = nodePtr.nonSurfIndex
        # # print(f"Extracted node {nodeIdx}")

        # if octreePtr.labelsPtr[0][nodeIdx] == 0:
        #     # has already been changed in a prev iter by its neighbour
        #     nodePtrQueue.pop()
        #     continue
        # changedIdxesPtr.push(nodeIdx)
        # # inner_initial_cost = computeCost(octreePtr)
        # # print('2step: inner_initial_cost', inner_initial_cost)
        # neg_cost_diff = computeCostDiff(octreePtr, nodeIdx) # simulate move
        # octreePtr.labelsPtr[0][nodeIdx] = 0 # move
        # # print('a neg_cost_diff', neg_cost_diff, computeCost(octreePtr)-inner_initial_cost, nodeIdx)
        # neighbours = getCodeFullNeighbours(nodePtr.codePtr[0])
        # # neighbours = getCodeNeighbours(nodePtr.codePtr[0])
        # # for j in range(neighbours.size()):
        # #     neigh_nodePtr = getNode(octreePtr, neighbours[j])
        # #     if neigh_nodePtr is NULL:
        # #         continue
        # #     # consider all neighbours in that direction, could be many children. Do this in a queue
        # #     innerNodePtrQueue.push(neigh_nodePtr)
        # #     while not innerNodePtrQueue.empty():
        # #         innerNodePtr = innerNodePtrQueue.front()
        # #         if not innerNodePtr.isLeaf:
        # #             for childPtr in innerNodePtr.childrenPtrArr:
        # #                 if childPtr is not NULL:
        # #                     innerNodePtrQueue.push(childPtr)
        # #         else:
        # #             if innerNodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 1:
        # #                 # neighbour is inside
        # #                 neg_cost_diff += computeCostDiff(octreePtr, innerNodePtr.nonSurfIndex) # simulate move
        # #                 octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] = 0 # move
        # #                 # print('a neg_cost_diff', neg_cost_diff, computeCost(octreePtr)-inner_initial_cost, innerNodePtr.nonSurfIndex)
        # #                 changedIdxesPtr.push(innerNodePtr.nonSurfIndex)
        # #         innerNodePtrQueue.pop()
        # for j in range(neighbours.size()):
        #     neigh_nodePtr = getNode(octreePtr, neighbours[j])
        #     if neigh_nodePtr is NULL:
        #         continue
        #     # consider all neighbours in that direction, could be many children. Do this in a queue
        #     if neigh_nodePtr.isLeaf:
        #         if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
        #             # neighbour is inside
        #             neg_cost_diff += computeCostDiff(octreePtr, neigh_nodePtr.nonSurfIndex) # simulate move
        #             octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] = 0 # move
        #             neighbours2 = getCodeFullNeighbours(neigh_nodePtr.codePtr[0])
        #             changedIdxesPtr.push(neigh_nodePtr.nonSurfIndex)
        #             if three_step:
        #                 for k in range(neighbours2.size()):
        #                     neigh_nodePtr2 = getNode(octreePtr, neighbours2[k])
        #                     if neigh_nodePtr2 is NULL:
        #                         continue
        #                     innerNodePtrQueue.push(neigh_nodePtr2)
        #     else:
        #         innerNodePtrQueue.push(neigh_nodePtr) # could have inside children
        # while not innerNodePtrQueue.empty():
        #     innerNodePtr = innerNodePtrQueue.front()
        #     if not innerNodePtr.isLeaf:
        #         for childPtr in innerNodePtr.childrenPtrArr:
        #             if childPtr is not NULL:
        #                 innerNodePtrQueue.push(childPtr)
        #     else:
        #         if innerNodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 1:
        #             # neighbour is inside
        #             neg_cost_diff += computeCostDiff(octreePtr, innerNodePtr.nonSurfIndex) # simulate move
        #             octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] = 0 # move
        #             # print('a neg_cost_diff', neg_cost_diff, computeCost(octreePtr)-inner_initial_cost, innerNodePtr.nonSurfIndex)
        #             changedIdxesPtr.push(innerNodePtr.nonSurfIndex)
        #     innerNodePtrQueue.pop()
        # # if changedIdxesPtr.size() < (2 if octreePtr.is2D else 3):
        # if changedIdxesPtr.size() == 1:
        #     # undo full move and skip to next
        #     while not changedIdxesPtr.empty():
        #         octreePtr.labelsPtr[0][changedIdxesPtr.front()] = 1 # undo move
        #         changedIdxesPtr.pop()
        #     nodePtrQueue.pop()
        #     # print('removed')
        #     continue
        # inQueue[nodeIdx] = 0
        # # octreePtr.labelsPtr[0][nodeIdx] = 0 # try changing to outside
        # # new_cost = computeCost(octreePtr)
        # # neg_cost_diff += makeSingleMove(octreePtr, changedIdxesPtr)
        # # val2 = computeCost(octreePtr)
        # neg_cost_diff += makeSingleMove(octreePtr, changedIdxesPtr)
        # # print('sinlge move neg_cost_diff', neg_cost_diff, computeCost(octreePtr)-inner_initial_cost)
        # # neg_cost_diff += val
        # # print(neg_cost_diff, val, neg_cost_diff+val, computeCost(octreePtr) - val2)
        # # print('\t 2 step move making, queue size {}, cost={:.5f}, true_cost = {:.5f}'.format(nodePtrQueue.size(), neg_cost_diff, computeCost(octreePtr)-currentCost))
        # assert changedIdxesPtr.size() == 0, changedIdxesPtr.size()
        # # neg_cost_diff = computeCostDiff(octreePtr, nodeIdx)
        # # if new_cost > currentCost:
        # # sumVal = 0
        # # for i in range(octreePtr.labelsPtr.size()):
        # #     sumVal += octreePtr.labelsPtr[0][i]
        # # print(sumVal, 'after move')

        # inner_initial_cost = computeCost(octreePtr)
        nodePtr = nodePtrQueue.front()
        oldLabelsPtr = new vector[int] (octreePtr.labelsPtr[0])
        if octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] == 0:
            # has already been changed in a prev iter by its neighbour
            nodePtrQueue.pop()
            continue
        item = <NodeCost*> malloc(sizeof(NodeCost))
        item.nonSurfIndex = nodePtr.nonSurfIndex
        item.cost = -computeCostDiff(octreePtr, nodePtr.nonSurfIndex)
        innerNodeCostPtrQueue.push(item)
        # print('pushed item', item.cost, item.nonSurfIndex)
        innerInQueue[nodePtr.nonSurfIndex] = 1
        neg_cost_diff = 0.0

        min_cost = 0.0
        best_num_moves = 0
        
        # assert changedIdxesPtr.size() == 0 # remove!!

        # max heap, so cost is negated
        while not innerNodeCostPtrQueue.empty():
            if changedIdxesPtr.size() > best_num_moves + 100 or neg_cost_diff > 100:
                while not innerNodeCostPtrQueue.empty():
                    innerNodeCostPtrQueue.pop()
                break
            if changedIdxesPtr.size() > three_step:
                while not innerNodeCostPtrQueue.empty():
                    innerNodeCostPtrQueue.pop()
                break
            item = innerNodeCostPtrQueue.top()
            # print('top item', item.cost, item.nonSurfIndex)
            innerNodePtr = octreePtr.nonSurfaceLeavesPtr[0][item.nonSurfIndex]
            # assert innerNodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 1, (innerNodePtr.isNonSurfLeaf, octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex])
            if octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 0:
                innerNodeCostPtrQueue.pop()
                continue
            neg_cost_diff += computeCostDiff(octreePtr, innerNodePtr.nonSurfIndex) # simulate move
            octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] = 0 # move
            changedIdxesPtr.push_back(innerNodePtr.nonSurfIndex)
            # POP NOW SO THAT TOP DOESN'T CHANGE WHEN ADDING NEIGHBOURS!!
            innerNodeCostPtrQueue.pop()
            # print('popped item', item.cost, item.nonSurfIndex)
            if neg_cost_diff < min_cost:
                best_num_moves = changedIdxesPtr.size()
                min_cost = neg_cost_diff
                # print('\tchanged min_cost', min_cost, best_num_moves)

                # inner_final_cost = computeCost(octreePtr)
                # # print('2step: inner_final_cost', inner_final_cost)
                # inner_actual_cc = inner_final_cost - inner_initial_cost
                # # print("sss", neg_cost_diff, inner_actual_cc, num_changed)
                # if abs(inner_actual_cc - neg_cost_diff) > 1e-4:
                #     print('###### inner current min', neg_cost_diff, inner_actual_cc)
            neighbours = getCodeNeighbours(innerNodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isLeaf:
                    if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
                        # neighbour is inside leaf
                        if innerInQueue[neigh_nodePtr.nonSurfIndex] == 0 or True: # isn't in the queue yet

                            item = <NodeCost*> malloc(sizeof(NodeCost))
                            item.nonSurfIndex = neigh_nodePtr.nonSurfIndex
                            item.cost = -computeCostDiff(octreePtr, neigh_nodePtr.nonSurfIndex)
                            innerNodeCostPtrQueue.push(item)
                            # print('pushed item', item.cost, item.nonSurfIndex)
                            innerInQueue[neigh_nodePtr.nonSurfIndex] = 1
        
        if best_num_moves != 0:
            print('best cost', min_cost, 'best num moves', best_num_moves, 'tried num_moves', changedIdxesPtr.size())
        while changedIdxesPtr.size() != best_num_moves:
            octreePtr.labelsPtr[0][changedIdxesPtr.back()] = 1 # undo move
            innerInQueue[changedIdxesPtr.back()] = 0 # remove from innerInQueue
            changedIdxesPtr.pop_back()
        neg_cost_diff = min_cost
        # now just clean up the rest
        while changedIdxesPtr.size() != 0:
            innerInQueue[changedIdxesPtr.back()] = 0 # remove from innerInQueue
            changedIdxesPtr.pop_back()
            

        # changedIdxesPtr.push(nodePtr.nonSurfIndex)
        # # inner_initial_cost = computeCost(octreePtr)
        # # print('2step: inner_initial_cost', inner_initial_cost)
        # neg_cost_diff = computeCostDiff(octreePtr, nodePtr.nonSurfIndex) # simulate move
        # octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0 # move
        # # print('a neg_cost_diff', neg_cost_diff, computeCost(octreePtr)-inner_initial_cost, nodePtr.nonSurfIndex)
        # neighbours = getCodeNeighbours(nodePtr.codePtr[0]) # not FULL neighbours


        # for j in range(neighbours.size()):
        #     neigh_nodePtr = getNode(octreePtr, neighbours[j])
        #     if neigh_nodePtr is NULL:
        #         continue
        #     # consider all neighbours in that direction, could be many children. Do this in a queue
        #     if neigh_nodePtr.isLeaf:
        #         if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
        #             # neighbour is inside
        #             neg_cost_diff += computeCostDiff(octreePtr, neigh_nodePtr.nonSurfIndex) # simulate move
        #             octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] = 0 # move
        #             neighbours2 = getCodeFullNeighbours(neigh_nodePtr.codePtr[0])




        if neg_cost_diff >= 0:
            # reject move and change back
            # octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 1
            # free(octreePtr.labelsPtr)
            # octreePtr.labelsPtr = oldLabelsPtr
            octreePtr.labelsPtr[0] = move(oldLabelsPtr[0])
            free(oldLabelsPtr)
            # sumVal = 0
            # for i in range(octreePtr.labelsPtr.size()):
            #     sumVal += octreePtr.labelsPtr[0][i]
            # print(sumVal, 'move rejected')
            neg_cost_diff = 0.0
        else:
            # octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0
            free(oldLabelsPtr)
            # sumVal = 0
            # for i in range(octreePtr.labelsPtr.size()):
            #     sumVal += octreePtr.labelsPtr[0][i]
            # print(sumVal, 'move accepted')
            # print("accepted", currentCost, new_cost, nodePtrQueue.size())
            # print("accepted", currentCost, nodePtrQueue.size())
            # # accept move and update current cost
            # currentCost = new_cost
            costChange += neg_cost_diff
            num_changed += 1
            # also add new neighbours
            # neighbours = getCodeFullNeighbours(nodePtr.codePtr[0])
            # for j in range(neighbours.size()):
            #     neigh_nodePtr = getNode(octreePtr, neighbours[j])
            #     if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 1:
            #         # neighbour is inside
            #         if not inQueue[neigh_nodePtr.nonSurfIndex]:
            #             nodePtrQueue.push(neigh_nodePtr)
            #             inQueue[neigh_nodePtr.nonSurfIndex] = 1
            #             # print(f'added node {neigh_nodePtr.nonSurfIndex}')
        nodePtrQueue.pop()
        # inner_final_cost = computeCost(octreePtr)
        # # print('2step: inner_final_cost', inner_final_cost)
        # inner_actual_cc = inner_final_cost - inner_initial_cost
        # # print("sss", neg_cost_diff, inner_actual_cc, num_changed)
        # if abs(inner_actual_cc - neg_cost_diff) > 1e-4:
        #     print('###### inner', neg_cost_diff, inner_actual_cc)
    print(f"At the end of 2 step move making, groups changed={num_changed}")
    # cdef float final_cost = computeCost(octreePtr)
    # print('2step: final_cost', final_cost)
    # cdef float actual_cc = final_cost - initial_cost
    # # print("Cost change = {:.5f} / {:.5f}".format(costChange, actual_cc))
    # if abs(actual_cc - costChange) > 1e-3:
    #     print('######', costChange, actual_cc)
    return num_changed



cdef void grow_if_cost_reduce(Octree* octreePtr):
    cdef queue[OTNode*] nodePtrQueue
    cdef Py_ssize_t i, j
    cdef int num_changed = 0
    cdef int count
    cdef queue[OTNode*] innerNodePtrQueue

    for i in range(octreePtr.labelsPtr.size()):
        if octreePtr.labelsPtr[0][i] == 1:
            # inside
            nodePtr = octreePtr.nonSurfaceLeavesPtr[0][i]
            neighbours = getCodeNeighbours(nodePtr.codePtr[0])
            for j in range(neighbours.size()):
                neigh_nodePtr = getNode(octreePtr, neighbours[j])
                if neigh_nodePtr is NULL:
                    continue
                if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                    # neighbour is outside
                    nodePtrQueue.push(nodePtr)
    # print("queue len:", nodePtrQueue.size())
    while not nodePtrQueue.empty():
        # print('size', nodePtrQueue.size())
        nodePtr = nodePtrQueue.front()
        # neighbours = getCodeNeighbours(nodePtr.codePtr[0])
        neighbours = getCodeFullNeighbours(nodePtr.codePtr[0])
        if octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] == 0:
            # already changed to outside
            nodePtrQueue.pop()
            continue
        count = 0
        for i in range(neighbours.size()):
            neigh_nodePtr = getNode(octreePtr, neighbours[i])
            if neigh_nodePtr is NULL:
                continue
            if neigh_nodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr.nonSurfIndex] == 0:
                # inside node have outside neighbour
                count += 1
            if count == (2 if octreePtr.is2D else 3):
            # if count == (4 if octreePtr.is2D else 8):
            # if count == (5 if octreePtr.is2D else 14):
                # set to outside
                octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0

                neg_cost_diff = computeCostDiff(octreePtr, nodePtr.nonSurfIndex)
                if neg_cost_diff >= 0:
                    # reject move and change back
                    octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 1
                else:
                    octreePtr.labelsPtr[0][nodePtr.nonSurfIndex] = 0
                    # costChange += neg_cost_diff
                    num_changed += 1

                    # now add all its inside neighbours to the queue
                    # neighbours = getCodeNeighbours(neigh_node.codePtr[0])
                    for j in range(neighbours.size()):
                        # neigh_nodePtr2 = getNode(octreePtr, neighbours[j])
                        # # if not neigh_nodePtr2.isLeaf:                    
                        # if neigh_nodePtr2.isNonSurfLeaf and octreePtr.labelsPtr[0][neigh_nodePtr2.nonSurfIndex] == 1:
                        #     # neighbour is inside
                        #     nodePtrQueue.push(neigh_nodePtr2)
                        neigh_nodePtr = getNode(octreePtr, neighbours[j])
                        if neigh_nodePtr is NULL:
                            continue
                        innerNodePtrQueue.push(neigh_nodePtr)
                        while not innerNodePtrQueue.empty():
                            innerNodePtr = innerNodePtrQueue.front()
                            # if innerNodePtr is NULL:
                            #     innerNodePtrQueue.pop()
                            #     continue
                            if not innerNodePtr.isLeaf:
                                for childPtr in innerNodePtr.childrenPtrArr:
                                    if childPtr is not NULL:
                                        innerNodePtrQueue.push(childPtr)
                            else:
                                if innerNodePtr.isNonSurfLeaf and octreePtr.labelsPtr[0][innerNodePtr.nonSurfIndex] == 1:
                                    # Neighbour (or child of neighbour which might not be a neighbour) is inside
                                    nodePtrQueue.push(innerNodePtr)
                            innerNodePtrQueue.pop()
                break
        nodePtrQueue.pop()
    print(f"At the end of grow, num_changed={num_changed}")
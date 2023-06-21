# distutils: language=c++
#cython: language_level=3
from __future__ import print_function
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp cimport bool


#####################################################################
### Octree Node struct, OTNode, and its Python Wrapper, PyOTNode
#####################################################################

ctypedef struct OTNode:
    int depth
    float length
    float top_pos_x, top_pos_y, top_pos_z
    OTNode* parentPtr
    int isLeaf
    int isNonSurfLeaf
    int nonSurfIndex
    OTNode* childrenPtrArr[8]
    vector[Py_ssize_t]* codePtr
    vector[Py_ssize_t]* index3dPtr
    vector[Py_ssize_t]* pointIdxesPtr
    vector[NodeDist]* closestNonSurfIdxesPtr
    vector[int]* relevantSurfIdxesPtr
    int _tempLabel
    bool is2D

cdef void initialiseOTNodePointer(OTNode *_ptr):
    if _ptr is NULL:
        raise MemoryError
    _ptr.depth = 0
    _ptr.length = 0.0
    _ptr.top_pos_x = 0.0
    _ptr.top_pos_y = 0.0
    _ptr.top_pos_z = 0.0
    _ptr.isLeaf = 1
    _ptr.isNonSurfLeaf = 0
    _ptr.nonSurfIndex = 0
    _ptr.parentPtr = NULL
    _ptr.codePtr = NULL
    _ptr.index3dPtr = NULL
    _ptr.pointIdxesPtr = NULL
    _ptr.closestNonSurfIdxesPtr = NULL
    _ptr.relevantSurfIdxesPtr = NULL
    _ptr.childrenPtrArr = [NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL]
    _ptr._tempLabel = 1
    _ptr.is2D = False

cdef class PyOTNode:
    # cdef OTNode *_ptr         # Will be made
    # cdef bint ptr_owner       # Will be made

    def __cinit__(self):
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    # Extension class properties
    @property
    def depth(self):
        return self._ptr.depth if self._ptr is not NULL else None
    @property
    def length(self):
        return self._ptr.length if self._ptr is not NULL else None
    @property
    def top_pos_x(self):
        return self._ptr.top_pos_x if self._ptr is not NULL else None
    @property
    def top_pos_y(self):
        return self._ptr.top_pos_y if self._ptr is not NULL else None
    @property
    def top_pos_z(self):
        return self._ptr.top_pos_z if self._ptr is not NULL else None
    @property
    def isLeaf(self):
        return self._ptr.isLeaf if self._ptr is not NULL else None
    @property
    def isNonSurfLeaf(self):
        return self._ptr.isNonSurfLeaf if self._ptr is not NULL else None
    @property
    def nonSurfIndex(self):
        return self._ptr.nonSurfIndex if self._ptr is not NULL else None
    @property
    def code(self):
        if self._ptr is NULL:
            None
        else:
            return np.asarray(self._ptr.codePtr[0], dtype=np.int32) if self._ptr.codePtr is not NULL else None
    @property
    def index3d(self):
        if self._ptr is NULL:
            None
        else:
            return np.asarray(self._ptr.index3dPtr[0], dtype=np.int32) if self._ptr.index3dPtr is not NULL else None
    @property
    def pointIdxes(self):
        if self._ptr is NULL:
            None
        else:
            return np.asarray(self._ptr.pointIdxesPtr[0], dtype=np.int32)
    @property
    def closestNonSurfIdxes(self):
        cdef Py_ssize_t i
        if self._ptr is NULL:
            None
        else:
            ret_lst = []
            for i in range(self._ptr.closestNonSurfIdxesPtr.size()):
                ret_lst.append((self._ptr.closestNonSurfIdxesPtr[0][i].nonSurfIndex, self._ptr.closestNonSurfIdxesPtr[0][i].dist))
            return ret_lst
    @property
    def relevantSurfIdxes(self):
        if self._ptr is NULL:
            None
        else:
            return np.asarray(self._ptr.relevantSurfIdxesPtr[0], dtype=np.int32)
    @property
    def parent(self):
        if self._ptr is NULL:
            return None
        else:
            return PyOTNode.from_ptr(self._ptr.parentPtr, owner=False)
    @property
    def children(self):
        if self._ptr is NULL or self.isLeaf:
            return None
        else:
            children_lst = []
            if self.is2D:
                for i in range(4):
                    child = PyOTNode.from_ptr(self._ptr.childrenPtrArr[i], owner=False)
                    children_lst.append(child)
            else:
                for i in range(8):
                    child = PyOTNode.from_ptr(self._ptr.childrenPtrArr[i], owner=False)
                    if self._ptr.childrenPtrArr[i] is not NULL:
                        children_lst.append(child)
            return children_lst
    @property
    def is2D(self):
        return self._ptr.is2D if self._ptr is not NULL else None
    

    def __getitem__(self,index):
        assert isinstance(index, (int, np.int32)), (index,type(index))
        if self.is2D:
            assert 0<= index <= 3, index
        else:
            assert 0<= index <= 7, index
        # assert not self.is_leaf(), f"{self} is a leaf" # raise IndexError?
        if self._ptr is NULL or self.isLeaf:
            return None
        else:
            return PyOTNode.from_ptr(self._ptr.childrenPtrArr[index], owner=False)
    
    def __getstate__(self):
        return (self.depth, self.length, self.top_pos_x, self.top_pos_y, self.top_pos_z,self.isLeaf, self.isNonSurfLeaf, 
                    self.nonSurfIndex, self._tempLabel, self.is2D)
    
    def __setstate__(self, depth, length, top_pos_x, top_pos_y, top_pos_z, isLeaf, isNonSurfLeaf, nonSurfIndex, _tempLabel, is2D):
        self._ptr.depth = depth
        self._ptr.length = length
        self._ptr.top_pos_x = top_pos_x
        self._ptr.top_pos_y = top_pos_y
        self._ptr.top_pos_z = top_pos_z
        self._ptr.isLeaf = isLeaf
        self._ptr.isNonSurfLeaf = isNonSurfLeaf
        self._ptr.nonSurfIndex = nonSurfIndex
        self._ptr._tempLabel = _tempLabel
        self._ptr.is2D = is2D
    
    def __reduce__(self):
        return (rebuild_PyOTNode, self.__getstate__())

    @staticmethod
    cdef PyOTNode from_ptr(OTNode *_ptr, bint owner=False):
        """Factory function to create PyOTNode objects from
        given OTNode pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef PyOTNode wrapper = PyOTNode.__new__(PyOTNode)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


    @staticmethod
    cdef PyOTNode new_struct():
        """Factory function to create PyOTNode objects with
        newly allocated OTNode"""
        cdef OTNode *_ptr = <OTNode *>malloc(sizeof(OTNode))

        initialiseOTNodePointer(_ptr)

        return PyOTNode.from_ptr(_ptr, owner=True)

def rebuild_PyOTNode(*args):
    py_node = PyOTNode.new_struct()
    py_node.__setstate__(*args)
    return py_node


#####################################################################
### Octree struct, OTNode, and its Python Wrapper, PyOctree
#####################################################################

ctypedef struct Octree:
    int max_depth
    OTNode* rootPtr
    vector[OTNode*]* surfaceLeavesPtr
    vector[OTNode*]* nonSurfaceLeavesPtr
    vector[int]* labelsPtr
    bool is2D

cdef void initialiseOctreePointer(Octree *_ptr):
    if _ptr is NULL:
        raise MemoryError
    _ptr.max_depth = 0
    _ptr.is2D = False


cdef vector[Py_ssize_t] index3d2code(vector[Py_ssize_t] index3d, int depth) nogil:
    # depth starts from 0, so should be depth + 1
    cdef vector[Py_ssize_t] code = vector[Py_ssize_t](depth+1, 0)
    cdef Py_ssize_t i, j
    for i in range(3):
        for j in range(depth): # not depth + 1 here as the first value is always 0 anyway
            code[depth-j] += ((index3d[2-i] >> j) & 1) * 2 ** i
    # print(code, index3d)
    return code

cdef OTNode* getNode(Octree* octreePtr, vector[Py_ssize_t] code) nogil:
    # assert code.size() > 0
    # assert code[0] == 0
    cdef Py_ssize_t i
    cdef OTNode* nodePtr = octreePtr.rootPtr
    for i in range(1, code.size()):
        if nodePtr.isLeaf:
            break
        nodePtr = nodePtr.childrenPtrArr[code[i]]
    return nodePtr

cdef class PyOctree:
    # cdef Octree *_ptr         # Will be made
    # cdef bint ptr_owner       # Will be made

    # def __cinit__(self):
    #     self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    # Extension class properties
    
    @property
    def max_depth(self):
        return self._ptr.max_depth if self._ptr is not NULL else None
    @property
    def root(self):
        if self._ptr is NULL:
            return None
        else:
            return PyOTNode.from_ptr(self._ptr.rootPtr, owner=False)
    @property
    def labels(self):
        if self._ptr is NULL:
            None
        else:
            return np.asarray(self._ptr.labelsPtr[0], dtype=np.int32)
    
    def set_labels(self, labels):
        # print('in set labels')
        # print(self._ptr.labelsPtr.size(), labels.shape)
        for i in range(self._ptr.labelsPtr.size()):
            self._ptr.labelsPtr[0][i] = labels[i]
    @property
    def numSurfaceLeaves(self):
        if self._ptr is NULL:
            None
        else:
            return self._ptr.surfaceLeavesPtr.size()
    @property
    def numNonSurfaceLeaves(self):
        if self._ptr is NULL:
            None
        else:
            return self._ptr.nonSurfaceLeavesPtr.size()
    @property
    def surfaceLeaves(self):
        if self._ptr is NULL:
            None
        else:
            return (PyOTNode.from_ptr(self._ptr.surfaceLeavesPtr[0][i], owner=False) 
                                for i in range(self._ptr.surfaceLeavesPtr.size()))
    @property
    def nonSurfaceLeaves(self):
        if self._ptr is NULL:
            None
        else:
            return (PyOTNode.from_ptr(self._ptr.nonSurfaceLeavesPtr[0][i], owner=False) 
                                for i in range(self._ptr.nonSurfaceLeavesPtr.size()))
    @property
    def labels(self):
        if self._ptr is NULL:
            None
        else:
            return np.asarray(self._ptr.labelsPtr[0], dtype=np.int32)
    @property
    def is2D(self):
        return self._ptr.is2D if self._ptr is not NULL else None
    
    def __getstate__(self):
        return (self.max_depth, self.root, self.surfaceLeaves, self.nonSurfaceLeaves, self.labels, self.is2D)
    
    def __setstate__(self, max_depth, root, surfaceLeaves, nonSurfaceLeaves, labels, is2D):
        self._ptr.max_depth = max_depth
        self._ptr.rootPtr = (<PyOTNode>root)._ptr

        self._ptr.surfaceLeavesPtr = new vector[OTNode*] ()
        for node in surfaceLeaves:
           self._ptr.surfaceLeavesPtr.push_back((<PyOTNode>node)._ptr)
        
        self._ptr.nonSurfaceLeavesPtr = new vector[OTNode*] ()
        for node in nonSurfaceLeaves:
           self._ptr.nonSurfaceLeavesPtr.push_back((<PyOTNode>node)._ptr)
        
        self._ptr.labelsPtr = new vector[int] ()
        for label in labels:
            self._ptr.labelsPtr.push_back(label)
        
        self._ptr.is2D = is2D
    
    def __reduce__(self):
        return (rebuild_PyOctree, self.__getstate__())
    
    def nodeFromIndex3d(self, py_index3d, depth):
        cdef vector[Py_ssize_t] index3d = vector[Py_ssize_t]()
        for num in py_index3d:
            index3d.push_back(num)
        code = index3d2code(index3d, depth)
        nodePtr = getNode(self._ptr, code)
        return PyOTNode.from_ptr(nodePtr, owner=False)

    cdef OTNode* nodePtrFromIndex3d(self, int[:] py_index3d, int depth):
        cdef vector[Py_ssize_t] index3d = vector[Py_ssize_t]()
        cdef Py_ssize_t i
        cdef Py_ssize_t num = py_index3d.shape[0]
        for i in range(num):
            index3d.push_back(py_index3d[i])
        code = index3d2code(index3d, depth)
        return getNode(self._ptr, code)
    
    def signsFromIndex3ds(self, int[:,:] py_index3ds, int depth):
        cdef Py_ssize_t i
        cdef Py_ssize_t num = py_index3ds.shape[0]
        cdef OTNode* nodePtr
        signs2 = []
        for i in range(num):
            if i % 5000000 == 0:
                print("{}/{}".format(i, num))
            nodePtr = self.nodePtrFromIndex3d(py_index3ds[i], depth)
            if nodePtr is NULL:
                print('NULL')
            else:
                if nodePtr.isNonSurfLeaf:
                    if self._ptr.labelsPtr[0][nodePtr.nonSurfIndex]:
                        signs2.append(-1)
                    else:
                        signs2.append(1)
                else:
                    signs2.append(-1)
        return signs2
    

    @staticmethod
    cdef PyOctree from_ptr(Octree *_ptr, bint owner=False):
        """Factory function to create PyOctree objects from
        given Octree pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef PyOctree wrapper = PyOctree.__new__(PyOctree)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


    @staticmethod
    cdef PyOctree new_struct():
        """Factory function to create PyOctree objects with
        newly allocated Octree"""
        cdef Octree *_ptr = <Octree *>malloc(sizeof(Octree))

        initialiseOctreePointer(_ptr)

        return PyOctree.from_ptr(_ptr, owner=True)

def rebuild_PyOctree(*args):
    py_ot = PyOctree.new_struct()
    py_ot.__setstate__(*args)
    return py_ot


#####################################################################
### Structs for data about nodes:
### - Distance to a node: NodeDist
### - < comprison on NodeDist pointers: NodeDistPtrLess
### - Priority Queue on NodeDist pointers: NodeDistQueue
### - Cost of a node: NodeCost
### - < comprison on NodeCost pointers: NodeCostPtrLess
### - Priority Queue on NodeCost pointers: NodeCostQueue
#####################################################################


ctypedef struct NodeDist:
    int nonSurfIndex
    float dist 

ctypedef struct NodeCost:
    int nonSurfIndex
    float cost






# distutils: language=c++
#cython: language_level=3
cimport cython
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

cdef void initialiseOTNodePointer(OTNode *_ptr)

cdef class PyOTNode:
    cdef OTNode *_ptr
    cdef bint ptr_owner

    @staticmethod
    cdef PyOTNode from_ptr(OTNode *_ptr, bint owner=*)

    @staticmethod
    cdef PyOTNode new_struct()

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

cdef void initialiseOctreePointer(Octree *_ptr)

cdef class PyOctree:
    cdef Octree *_ptr
    cdef bint ptr_owner

    # cpdef set_labels(self, int[:] labels)

    cdef OTNode* nodePtrFromIndex3d(self, int[:] py_index3d, int depth)

    @staticmethod
    cdef PyOctree from_ptr(Octree *_ptr, bint owner=*)

    @staticmethod
    cdef PyOctree new_struct()

    # @classmethod
    # cdef void initialisePointer(Octree *_ptr)



#####################################################################
### Structs for data about nodes:
### - Distance to a node: NodeDist
### - < comprison on NodeDist pointers: NodeDistPtrLess
### - Priority Queue on NodeDist pointers: NodeDistQueue
### - Cost of a node: NodeCost
### - < comprison on NodeCost pointers: NodeCostPtrLess
### - Priority Queue on NodeCost pointers: NodeCostQueue
#####################################################################

cdef extern from *:
    """
    #include <queue>
    using namespace std;
    typedef struct {
        float dist;  
        int nonSurfIndex;    
    } NodeDist;

    struct NodeDistPtrLess {
        bool operator()(const NodeDist* a, const NodeDist* b) {
            return a->dist < b->dist;
        }
    };

    typedef std::priority_queue<NodeDist*, std::vector<NodeDist*>, NodeDistPtrLess> NodeDistQueue;
    """
    ctypedef struct NodeDist:
        float dist
        int nonSurfIndex
    
    #using cpp_pq = std::priority_queue<T,std::vector<T>,std::function<bool(T,T)>>;
    cdef cppclass NodeDistQueue:
            NodeDistQueue(...) nogil except +
            void push(NodeDist*) nogil
            NodeDist* top() nogil
            void pop() nogil
            bool empty() nogil
            int size() nogil

cdef extern from *:
    """
    #include <queue>
    using namespace std;
    typedef struct {
        float cost;  
        int nonSurfIndex;    
    } NodeCost;

    struct NodeCostPtrLess {
        bool operator()(const NodeCost* a, const NodeCost* b) {
            return a->cost < b->cost;
        }
    };

    typedef std::priority_queue<NodeCost*, std::vector<NodeCost*>, NodeCostPtrLess> NodeCostQueue;
    """
    ctypedef struct NodeCost:
        float cost
        int nonSurfIndex
    
    #using cpp_pq = std::priority_queue<T,std::vector<T>,std::function<bool(T,T)>>;
    cdef cppclass NodeCostQueue:
            NodeCostQueue(...) nogil except +
            void push(NodeCost*) nogil
            NodeCost* top() nogil
            void pop() nogil
            bool empty() nogil
            int size() nogil
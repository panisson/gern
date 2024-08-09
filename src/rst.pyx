#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import cython
import numpy as np

cimport numpy as np
from cpython cimport array as arr
from libc.stdlib cimport srand, rand, RAND_MAX

ctypedef np.uint8_t BTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.uint64_t UITYPE_t

# Returns an integer in the range [0, n).
# Uses rand(), and so is affected-by/affects the same seed.

@cython.cdivision(True)
@cython.nonecheck(False)
cdef inline UITYPE_t randint(UITYPE_t n) noexcept:
    # Chop off all of the values that would cause skew...
    cdef int end = RAND_MAX / n # truncate skew
    end *= n
    
    # ... and ignore results from rand() that fall above that limit.
    # (Worst case the loop condition should succeed 50% of the time,
    # so we can expect to bail out of this loop pretty quickly.)
    cdef int r = rand()
    while r >= end:
        r = rand()
    return r % n

def random_spanning_tree(csgraph, seed):
    'Returns a uniform random spanning tree of graph'
    
    cdef UITYPE_t start_vtx, current_vtx
    cdef UITYPE_t first_vtx, next_vtx
    cdef UITYPE_t s, t
    cdef UITYPE_t i
    cdef UITYPE_t visited_idx, num_visited
    cdef UITYPE_t N
    
    cdef list path = list()
    
    cdef np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag
    cdef np.ndarray[ITYPE_t, ndim=1, mode='c'] visit_order
    cdef np.ndarray[UITYPE_t, ndim=2, mode='c'] rst
    cdef UITYPE_t num_rst_edges
    
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = csgraph.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = csgraph.indices
    cdef int num_neighs, indptr_start, indptr_end, next_idx
    
    visited_idx = 0
    num_visited = 0
    N = csgraph.shape[0]
    
    np.random.seed(seed)
    srand(seed)
    
    visit_order = np.arange(N)
    np.random.shuffle(visit_order)
    unvisited_flag = np.ones(N, dtype=np.uint8)
    rst = np.zeros((2, N-1), dtype=np.uint64)
    num_rst_edges = 0
    
    first_vtx = visit_order[visited_idx]
    visited_idx += 1
    num_visited += 1
    unvisited_flag[first_vtx] = 0
    
    while num_visited < N:
        
        start_vtx = visit_order[visited_idx]
        visited_idx += 1
        while not unvisited_flag[start_vtx]:
            start_vtx = visit_order[visited_idx]
            visited_idx += 1
        
        current_vtx = start_vtx
        path.clear()
        path.append(current_vtx)
        while unvisited_flag[current_vtx]:
            
            indptr_start = indptr[current_vtx]
            indptr_end = indptr[current_vtx+1]

            num_neighs = indptr_end - indptr_start
            next_idx = randint(num_neighs)
            next_vtx = indices[indptr_start + next_idx]
            
            if next_vtx in path: 
                i = path.index(next_vtx) 
                path = path[:i]
            path.append(next_vtx)
            current_vtx = next_vtx

        s = path[0]
        for t in path[1:]:
            unvisited_flag[s] = 0
            num_visited += 1
            
            rst[0, num_rst_edges] = s
            rst[1, num_rst_edges] = t
            num_rst_edges += 1
            
            s = t
    
    return rst

def rw_random_spanning_tree(csgraph, int max_cycles=-1):
    """
    Returns a uniform random spanning tree of graph
    using the random walk method
    """
    cdef int N
    cdef int num_visited
    cdef int current_vtx, next_vtx
    cdef int cycles
    cdef np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag
    cdef int num_rst_edges
    cdef np.ndarray[UITYPE_t, ndim=2, mode='c'] rst
    cdef list neighs
    cdef int next_idx
    
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = csgraph.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = csgraph.indices
    cdef int num_neighs, indptr_start, indptr_end
    
    N = csgraph.shape[0]
    
    rst = np.zeros((2, N-1), dtype=np.uint64)
    
    unvisited_flag = np.ones(N, dtype=np.uint8)
    num_visited = 0
    num_rst_edges = 0
    
    current_vtx = randint(N)
    unvisited_flag[current_vtx] = 0
    num_visited += 1
    cycles = 0
    while num_visited < N:
        cycles += 1
        if max_cycles > 0 and cycles >= max_cycles:
            break
        
        indptr_start = indptr[current_vtx]
        indptr_end = indptr[current_vtx+1]
        
        num_neighs = indptr_end - indptr_start
        next_idx = randint(num_neighs)
        next_vtx = indices[indptr_start + next_idx]
        
        if unvisited_flag[next_vtx]:
            unvisited_flag[next_vtx] = 0
            num_visited += 1
            
            rst[0, num_rst_edges] = current_vtx
            rst[1, num_rst_edges] = next_vtx
            num_rst_edges += 1
            
        current_vtx = next_vtx
    return rst, num_rst_edges
    

def random_spanning_tree_v2(csgraph, int seed):
    'Returns a uniform random spanning tree of graph'
    
    cdef UITYPE_t start_vtx, current_vtx
    cdef UITYPE_t first_vtx, next_vtx
    cdef UITYPE_t s, t
    cdef UITYPE_t i
    cdef UITYPE_t visited_idx, num_visited
    cdef UITYPE_t N
    
    cdef arr.array path  = arr.array('I')
    cdef int j
    
    cdef np.ndarray[BTYPE_t, ndim=1, mode='c'] unvisited_flag
    cdef np.ndarray[ITYPE_t, ndim=1, mode='c'] visit_order
    cdef np.ndarray[UITYPE_t, ndim=2, mode='c'] rst
    cdef UITYPE_t num_rst_edges
    
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = csgraph.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = csgraph.indices
    cdef int num_neighs, indptr_start, indptr_end, next_idx
    
    visited_idx = 0
    num_visited = 0
    N = csgraph.shape[0]
    
    np.random.seed(seed)
    srand(seed)
    
    visit_order = np.arange(N)
    np.random.shuffle(visit_order)
    unvisited_flag = np.ones(N, dtype=np.uint8)
    
    rst, num_rst_edges = rw_random_spanning_tree(csgraph, max_cycles=N//4)
    assert num_rst_edges < N
    
    for i in range(num_rst_edges):
        s = rst[0, i]
        if unvisited_flag[s] == 1:
            unvisited_flag[s] = 0
            num_visited += 1
        
    for i in range(num_rst_edges):
        t = rst[1, i]
        if unvisited_flag[t] == 1:
            unvisited_flag[t] = 0
            num_visited += 1
    
    while num_visited < N:
        
        start_vtx = visit_order[visited_idx]
        visited_idx += 1
        while not unvisited_flag[start_vtx]:
            start_vtx = visit_order[visited_idx]
            visited_idx += 1
        
        current_vtx = start_vtx
        
        arr.resize_smart(path, 1)
        path.data.as_uints[0] = current_vtx
        
        while unvisited_flag[current_vtx]:
            
            indptr_start = indptr[current_vtx]
            indptr_end = indptr[current_vtx+1]

            num_neighs = indptr_end - indptr_start
            next_idx = randint(num_neighs)
            next_vtx = indices[indptr_start + next_idx]
            
            for j in range(len(path)):
                if path.data.as_uints[j] == next_vtx:
                    arr.resize_smart(path, j)
                    break
            
            i = len(path)
            arr.resize_smart(path, i+1)
            path.data.as_uints[i] = next_vtx
            
            current_vtx = next_vtx
            
        s = path.data.as_uints[0]
        for j in range(1, len(path)):
            t = path.data.as_uints[j]
            unvisited_flag[s] = 0
            num_visited += 1
            
            rst[0, num_rst_edges] = s
            rst[1, num_rst_edges] = t
            num_rst_edges += 1
            
            s = t
    
    assert num_rst_edges == N-1, (num_rst_edges, N-1)
    
    return rst
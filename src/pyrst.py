"""
Pure Python code for generating Uniform Random Spanning Trees (RSTs).
"""
import random
import numpy as np

def _rw_rst(N: int, indices, indptr, unvisited_flag, max_cycles=-1):
    
    num_unvisited = N
    rst = []
    current_vtx = random.randint(0, N-1)
    unvisited_flag[current_vtx] = 0
    num_unvisited -= 1
    cycles = 0
    while num_unvisited > 0:
        cycles += 1
        if max_cycles > 0 and cycles >= max_cycles:
            break
        neighs = indices[indptr[current_vtx]: indptr[current_vtx+1]]
        next_vtx = random.choice(neighs)
        if unvisited_flag[next_vtx]:
            rst.append((current_vtx, next_vtx))
            unvisited_flag[next_vtx] = 0
            num_unvisited -= 1
        current_vtx = next_vtx
    return rst

def _wilson_rst(N: int, indices, indptr, rst, unvisited_flag):
    num_visited = N - unvisited_flag.sum()
    
    visit_order = np.arange(N)
    np.random.shuffle(visit_order)
    first_vtx = visit_order[0]
    unvisited_flag[first_vtx] = 0
    num_visited += 1
    visited_idx = 1

    while num_visited < N:
        start_vtx = visit_order[visited_idx]
        visited_idx += 1
        while unvisited_flag[start_vtx] == 0:
            start_vtx = visit_order[visited_idx]
            visited_idx += 1

        current_vtx = start_vtx
        path = [current_vtx]
        while unvisited_flag[current_vtx]:

            neighs = indices[indptr[current_vtx]: indptr[current_vtx+1]]
            next_vtx = random.choice(neighs)
            if next_vtx in path: 
                i = path.index(next_vtx) 
                path = path[:i]
            path.append(next_vtx)
            current_vtx = next_vtx

        s = path[0]
        for t in path[1:]:
            rst.append((s, t))
            unvisited_flag[s] = 0
            num_visited += 1
            s = t
    return rst

def wilson_random_spanning_tree(csgraph, seed=-1):
    """
    Returns a uniform random spanning tree of graph
    using the Wilson algorithm.
    """
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)

    indices = csgraph.indices
    indptr = csgraph.indptr
    N = csgraph.shape[0]

    unvisited_flag = np.ones(N, dtype=bool)
    tree = []
    tree = _wilson_rst(N, indices, indptr, tree, unvisited_flag)
    return np.array(tree).T

def rw_random_spanning_tree(csgraph, seed=-1):
    """
    Returns a uniform random spanning tree of graph
    using the random walk method.
    """
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)

    indices = csgraph.indices
    indptr = csgraph.indptr
    N = csgraph.shape[0]
    unvisited_flag = np.ones(N, dtype=bool)

    tree = _rw_rst(N, indices, indptr, unvisited_flag, -1)
    return np.array(tree).T

def mixed_random_spanning_tree(csgraph, seed=-1):
    """
    Returns a uniform random spanning tree of graph.
    This function is a hybrid approach combining the Random Walk and Wilson methods.
    It first runs the Random Walk method until n edges are traversed, 
    and then switches to the Wilson algorithm.
    """
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)

    indices = csgraph.indices
    indptr = csgraph.indptr
    N = csgraph.shape[0]
    unvisited_flag = np.ones(N, dtype=bool)
    rw_cycles = csgraph.shape[0]//4

    tree = _rw_rst(N, indices, indptr, unvisited_flag, rw_cycles)
    tree = _wilson_rst(N, indices, indptr, tree, unvisited_flag)
    return np.array(tree).T

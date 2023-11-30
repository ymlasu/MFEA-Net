import numpy as np

def find_surrounding(img, pt):
    '''
    Given a point, return a list of points that are surrounding of current points inside the image
    Sequence starts from top left.
    '''
    temp = [[pt[0]-1,pt[1]-1],
            [pt[0]-1,pt[1]],
            [pt[0]-1,pt[1]+1],
            [pt[0],pt[1]-1],
            [pt[0],pt[1]+1],
            [pt[0]+1,pt[1]-1],
            [pt[0]+1,pt[1]],
            [pt[0]+1,pt[1]+1]]
    surrounding = [0, 1, 2, 3, 4, 5, 6, 7]
    dict_surrounding = dict(zip(surrounding, temp))
    h, w = img.shape
    for i, p in enumerate(temp):
        if(p[0] < 0 or p[1] < 0 or p[0] >= h or p[1] >= w):
            del dict_surrounding[i]
    return dict_surrounding

def reorder_connection(neumann_conn):
    '''
    Given a list of connected nodes, write a function to let it to be head-tail connected. 
    The order of each connection can be changed. For example if the input is 
    [[1,2],[2,4],[6,5],[4,3],[3,5]], the output should be 
    [[1,2],[2,4],[4,3],[3,5],[5,6]]. The first element of input should keep unchanged. 

    Input: neumann_conn, a list with each element is a two-element sublist
    Output: corrected order neumann_conn 
    '''

    graph = {}
    for a, b in neumann_conn:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    visited = set()
    result = []

    def dfs(node):
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                result.append([node, neighbor])
                dfs(neighbor)

    # Start from the first node in the input
    start_node = neumann_conn[0][0]
    visited.add(start_node)
    dfs(start_node)

    return np.array(result)

def define_left_nodes(img):
    '''
    Define the left boundary, also need to define connection. Please note that 
    this complex function needs to be validated if used in other cases. 

    The node id is arranged by row-first approach
    '''
    h_img, w_img = img.shape
    left = np.zeros((h_img+1, w_img+1))
    left_conn = []
    for i in range(h_img):
        for j in range(w_img):
            if(img[i,j]==1): # if there is material in current pixel i, j
                surrounding = find_surrounding(img, [i,j]) # surrounding pixels
                for key in surrounding: 
                    pt = surrounding[key]
                    if(img[pt[0],pt[1]]==0):
                        if(key == 0):
                            left[i,j] = 1
                        if(key == 1):
                            left[i,j] = 1
                            left[i,j+1] = 1
                            left_conn.append([
                                i*(w_img+1)+j,i*(w_img+1)+j+1
                            ]) # generate a connection
                        if(key == 2):
                            left[i,j+1] = 1
                        if(key == 3):
                            left[i,j] = 1
                            left[i+1,j] = 1
                            left_conn.append([
                                i*(w_img+1)+j,(i+1)*(w_img+1)+j
                            ]) # generate a connection
                        if(key == 4):
                            left[i,j+1] = 1
                            left[i+1,j+1] = 1
                            left_conn.append([
                                i*(w_img+1)+j+1,(i+1)*(w_img+1)+j+1
                            ]) # generate a connection
                        if(key == 5):
                            left[i+1,j] = 1
                        if(key == 6):
                            left[i+1,j] = 1
                            left[i+1,j+1] = 1
                            left_conn.append([
                                (i+1)*(w_img+1)+j,(i+1)*(w_img+1)+j+1
                            ]) # generate a connection
                        if(key == 7):
                            left[i+1,j+1] = 1

    return [left], [reorder_connection(left_conn)]


def define_right_nodes(node_mask):
    '''
    Input is the node mask
    Define top/bottom/right boundaries and their connections
    '''
    node = np.zeros_like(node_mask)
    node[:,-1] = 1
    node = (node * node_mask)
    node_idx = np.where(node.reshape(-1) == 1)
    conn = np.stack((node_idx[0][:-1],node_idx[0][1:]), axis=-1)
    return [node], [reorder_connection(conn)]

def define_top_nodes(node_mask):
    '''
    Input is the node mask
    Define top/bottom/right boundaries and their connections
    '''
    node = np.zeros_like(node_mask)
    node[-1,:] = 1
    node = (node * node_mask)
    node_idx = np.where(node.reshape(-1) == 1)
    conn = np.stack((node_idx[0][:-1],node_idx[0][1:]), axis=-1)
    return [node], [reorder_connection(conn)]

def define_bottom_nodes(node_mask):
    '''
    Input is the node mask
    Define top/bottom/right boundaries and their connections
    '''
    node = np.zeros_like(node_mask)
    node[0,:] = 1
    node = (node * node_mask)
    node_idx = np.where(node.reshape(-1) == 1)
    conn = np.stack((node_idx[0][:-1],node_idx[0][1:]), axis=-1)
    return [node], [reorder_connection(conn)]
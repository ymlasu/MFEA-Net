'''
define_boundary_nodes(mask)

Given a binary mask image, output the boundary nodes and connections.
Output: [[top_idx, top_conn], 
         [bottom_idx, bottom_conns], 
         [left_idx, left_conn], 
         [right_idx, right_conn], 
         [curved_idx, curved_conn]]
'''

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

def define_curved_nodes(img):
    '''
    Define the curved boundary, also need to define connection. Please note that 
    this complex function needs to be validated if used in other cases. 

    The node id is arranged by row-first approach
    '''
    h_img, w_img = img.shape
    curved = np.zeros((h_img+1, w_img+1))
    curved_conn = []
    for i in range(h_img):
        for j in range(w_img):
            if(img[i,j]==1): # if there is material in current pixel i, j
                surrounding = find_surrounding(img, [i,j]) # surrounding pixels

                # if surrounding doesn't contain 1, it is bottom edge
                # if surrounding doesn't contain 6, it is top edge
                # if surrounding doesn't contain 3, it is left edge
                # if surrounding doesn't contain 4, it is right edge
                
                for key in surrounding: 
                    pt = surrounding[key]
                    if(img[pt[0],pt[1]]==0):
                        if(key == 0):
                            curved[i,j] = 1
                        if(key == 1):
                            curved[i,j] = 1
                            curved[i,j+1] = 1
                            curved_conn.append([
                                i*(w_img+1)+j,i*(w_img+1)+j+1
                            ]) # generate a connection
                        if(key == 2):
                            curved[i,j+1] = 1
                        if(key == 3):
                            curved[i,j] = 1
                            curved[i+1,j] = 1
                            curved_conn.append([
                                i*(w_img+1)+j,(i+1)*(w_img+1)+j
                            ]) # generate a connection
                        if(key == 4):
                            curved[i,j+1] = 1
                            curved[i+1,j+1] = 1
                            curved_conn.append([
                                i*(w_img+1)+j+1,(i+1)*(w_img+1)+j+1
                            ]) # generate a connection
                        if(key == 5):
                            curved[i+1,j] = 1
                        if(key == 6):
                            curved[i+1,j] = 1
                            curved[i+1,j+1] = 1
                            curved_conn.append([
                                (i+1)*(w_img+1)+j,(i+1)*(w_img+1)+j+1
                            ]) # generate a connection
                        if(key == 7):
                            curved[i+1,j+1] = 1

    if(curved_conn != []):
        curved_conn = reorder_connection(curved_conn)
    return [curved, curved_conn]

def add_edge_connection(img, key, idx, conn, i, j):
    '''
    Node pattern:
    0 | 1 | 2
    3 | X | 4
    5 | 6 | 7
    '''
    _, w_img = img.shape

    if(key == 1 or key == 6):
        if(j == 0):
            idx[i,j] = 1
            idx[i+1,j] = 1
            conn.append([i*(w_img+1)+j,(i+1)*(w_img+1)+j])
        else:
            idx[i,j+1] = 1
            idx[i+1,j+1] = 1
            conn.append([i*(w_img+1)+j+1,(i+1)*(w_img+1)+j+1])
    if(key == 3 or key == 4):
        if(i == 0):
            idx[i,j] = 1
            idx[i,j+1] = 1
            conn.append([i*(w_img+1)+j,i*(w_img+1)+j+1])
        else: 
            idx[i+1,j] = 1
            idx[i+1,j+1] = 1
            conn.append([(i+1)*(w_img+1)+j,(i+1)*(w_img+1)+j+1])


def define_edge_nodes(img):
    '''
    Define edges other than the curved nodes
    Node pattern:
    0 | 1 | 2
    3 | X | 4
    5 | 6 | 7
    '''
    
    # define the straight boundary edges
    h_img, w_img = img.shape

    top_idx = np.zeros((h_img+1, w_img+1))
    bottom_idx = np.zeros((h_img+1, w_img+1))
    left_idx = np.zeros((h_img+1, w_img+1))
    right_idx = np.zeros((h_img+1, w_img+1))

    conn_top = []
    conn_bottom = []
    conn_left = []
    conn_right = []

    for i in range(h_img):
        for j in range(w_img):
            if(img[i,j]==1): # if there is material in current pixel i, j
                surr1 = find_surrounding(img, [i,j]) # surrounding pixels
                for key in surr1: 
                    pt = surr1[key]
                    if(img[pt[0],pt[1]]==1):
                        surr2 = find_surrounding(img, [pt[0],pt[1]])
                        if((6 not in surr1) and (6 not in surr2)):
                            # if surrounding doesn't contain 6, it is top edge
                            add_edge_connection(img, key, top_idx, conn_top, i, j)
                            #print("top: ", conn_top)
                        if((1 not in surr1) and (1 not in surr2)):
                            # if surrounding doesn't contain 1, it is bottom edge
                            add_edge_connection(img, key, bottom_idx, conn_bottom, i, j)
                            #print("bottom: ", conn_bottom)
                        if((3 not in surr1) and (3 not in surr2)):
                            # if surrounding doesn't contain 3, it is left edge
                            add_edge_connection(img, key, left_idx, conn_left, i, j)
                            #print("left: ", conn_left)
                        if((4 not in surr1) and (4 not in surr2)):
                            # if surrounding doesn't contain 4, it is right edge
                            add_edge_connection(img, key, right_idx, conn_right, i, j)
                            #print("right: ", conn_right)
    
    if(conn_top != []):
        conn_top = reorder_connection(conn_top)
    if(conn_bottom != []):
        conn_bottom = reorder_connection(conn_bottom)
    if(conn_left != []):
        conn_left = reorder_connection(conn_left)
    if(conn_right != []):
        conn_right = reorder_connection(conn_right)
            
    return [[top_idx, conn_top], [bottom_idx, conn_bottom], 
            [left_idx, conn_left], [right_idx, conn_right]]


def define_boundary_nodes(img):
    boundary = define_edge_nodes(img)
    curved_edge = define_curved_nodes(img) # assume there is only one curved edge
    boundary.append(curved_edge)
    return boundary
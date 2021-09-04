"""
    This module contains patterns to be projected, for calibration and scan.
    Also contains the decode procedure of structure light (patterns projected for scanning).
"""

import math

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

import cv2
max_val = 255

def genboard_white(resolution = [480, 640], max_val=1.0):
    
    board = max_val * np.ones((resolution[0], resolution[1], 3))
    return board

def genboard_projmarkers(interval=15, pattern = [6,6], resolution=[480, 640], offset=[280, 160], max_val=1.0):
    """
        This method is to generate projected markers. 
        The top right 3 points for entire pattern.

        paras:
        interval: input pixel level interval between points. int
        pattern: input the whole pattern. list. [pattern_nrows, pattern_ncols]
        resolution: input projected img resolution. list. [nrows, ncols]
        offset: input the position of top left point on image. pixel level. list. 
                [pixel_col(x), pixel_row(y)]

        board: returned board with 3 markers to project
    """

    nrows = resolution[0]
    ncols = resolution[1]

    pattern_nr = pattern[0]
    pattern_nw = pattern[1]

    if pattern_nr < 2:
        print("genboard_projmarkers: Not enough points to project!")
        return None

    board = np.zeros((nrows, ncols, 3))

    row0 = np.linspace(offset[0], offset[0] + 2 * interval * 4, pattern_nr, endpoint=False)
    col0 = np.linspace(offset[1], offset[1] + 2 * interval * 4, pattern_nw, endpoint=False)


    cv2.circle(board, (int(row0[pattern_nr-1]), int(col0[0])), 3, (max_val, max_val, max_val), -1)
    cv2.circle(board, (int(row0[pattern_nr-2]), int(col0[0])), 3, (max_val, max_val, max_val), -1)
    cv2.circle(board, (int(row0[pattern_nr-1]), int(col0[1])), 3, (max_val, max_val, max_val), -1)

    return board

def genboard_projchecker(interval=15, pattern = [6,6], resolution=[480, 640], offset=[280, 160], max_val=1.0):
    """
        This method is to generate projected checker points. 
        The entire pattern.

        paras:
        interval: input pixel level interval between points. int
        pattern: input the whole pattern. list. [pattern_nrows, pattern_ncols]
        resolution: input projected img resolution. list. [nrows, ncols]
        offset: input the position of top left point on image. pixel level. list. 
                [pixel_col(x), pixel_row(y)]

        board: returned board with all checker points to project
    """

    nrows = resolution[0]
    ncols = resolution[1]

    pattern_nr = pattern[0]
    pattern_nw = pattern[1]

    if pattern_nr < 2:
        print("genboard_projchecker: Not enough points to project!")
        return None

    board = np.zeros((nrows, ncols, 3))

    row0 = np.linspace(offset[0], offset[0] + 2 * interval * 4, pattern_nr, endpoint=False)
    col0 = np.linspace(offset[1], offset[1] + 2 * interval * 4, pattern_nw, endpoint=False)
    rows, cols = np.meshgrid(row0, col0)

    for i in range(0, pattern_nr):
        for j in range(0, pattern_nw):
            pt = [int(rows[i][j]), int(cols[i][j])]
            cv2.circle(board, (pt[0], pt[1]), 3, (max_val, max_val, max_val), -1)
            
    return board

def genboard_binarycode(proj_resolution = [800, 1280]):
    '''
    '''
    nrows = proj_resolution[0]
    ncols = proj_resolution[1]

    P = []
    offset = np.zeros(2, dtype = np.int)
    N = -1
    for j in range(0, 2):
        if j == 0:
            # vertical line
            # using // directly floor the offset and the offset int type, convenient!
            N = np.ceil(np.log2(ncols))
            offset[j] = (np.exp2(N) - ncols) // 2
        else:
            # horizontal line
            N = np.ceil(np.log2(nrows))
            offset[j] = (np.exp2(N) - nrows) // 2

        N = N.astype(np.int)
        #print("j is {}, N is {}".format(j, N))

        P_temp = np.zeros((nrows, ncols, N), dtype = np.uint8)
        numbers = list(range(0, np.exp2(N).astype(np.uint)))
        # generate binary representation for contiguous number and take transpose
        B = [[int(j) for j in list(format(i, 'b').zfill(N))] for i in numbers]
        B = np.array(B)

        # store strip pattern
        # transpose and reshpae is to make sure the result is correct
        if j == 0:
            for i in range(0, N):
                P_temp[:, :, i] = np.tile(B[offset[j] + 0: offset[j] + ncols, i].T, (nrows, 1))
        else:
            for i in range(0, N):
                P_temp[:, :, i] = np.tile(B[offset[j] + 0: offset[j] + nrows, i].reshape(-1, 1), (1, ncols))

        P.append(P_temp.copy()*255)

    return P

def genboard_graycode(proj_resolution = [800, 1280]):

    nrows = proj_resolution[0]
    ncols = proj_resolution[1]

    P = []
    offset = np.zeros(2, dtype = np.int)
    N = -1
    for j in range(0, 2):
        if j == 0:
            # vertical line
            # using // directly floor the offset and the offset int type, convenient!
            N = np.ceil(np.log2(ncols))
            offset[j] = (np.exp2(N) - ncols) // 2
        else:
            # horizontal line
            N = np.ceil(np.log2(nrows))
            offset[j] = (np.exp2(N) - nrows) // 2

        N = N.astype(np.int)
#        print("j is {}, N is {}".format(j, N))

        P_temp = np.zeros((nrows, ncols, N), dtype = np.uint8)
        numbers = list(range(0, np.exp2(N).astype(np.uint)))
        # generate binary representation for contiguous number and take transpose
        B = [[int(j) for j in list(format(i, 'b').zfill(N))] for i in numbers]
        B = np.array(B)
        B_shape = B.shape

        # get gray code pattern
        B_p = np.zeros(B_shape, dtype = np.uint8)
        B_p[:, 1:] = B[:, 0:-1]
        G = np.logical_xor(B, B_p)*1*max_val

        # store strip pattern
        # transpose and reshpae is to make sure the result is correct
        if j == 0:
            for i in range(0, N):
                P_temp[:, :, i] = np.tile(G[offset[j] + 0: offset[j] + ncols, i].T, (nrows, 1))
                # print("Image is {}".format(P_temp[:, :, i]))
        else:
            for i in range(0, N):
                P_temp[:, :, i] = np.tile(G[offset[j] + 0: offset[j] + nrows, i].reshape(-1, 1), (1, ncols))
                # print("Image is {}".format(P_temp[:, :, i]))

        P.append(P_temp.copy())
    return P

def decode_binarycode(B):
    """
        input is a stack of images with binary code structured light
        shape of B is (number, nrows, ncols)
    """
    # get the shape of the input

    N = B.shape[0]
    nrows = B.shape[1]
    ncols = B.shape[2]

    D = np.zeros((nrows, ncols), dtype = np.double)

    # converting binary code to decimal
    for i in range(0, N):
        D = D + np.exp2(i) * B[N - 1 - i, :, :].astype(np.double)

    D = D + 1

    return D

def decode_graycode(G):
    """
        input is a list of images with gray code structured light
    """
    # get the shape of the input

    N = G.shape[0]
    nrows = G.shape[1]
    ncols = G.shape[2]

    indexes = np.where(G == max_val)
    G[indexes] = 1

    print("The value of max_val in G is {}".format(np.where(G==max_val)))

    # convert gray code to binary
    # link: http://en.wikipedia.org/wiki/Gray_code
    B = np.zeros((N, nrows, ncols), dtype = np.uint8)

    B[0, :, :] = G[0, :, :]

    for i in range(1, N):
        B[i, :, :] = np.logical_xor(B[i - 1, :, :], G[i, :, :]).astype(np.double)

    # converting binary code to decimal
    D = np.zeros((nrows, ncols), dtype = np.double)

    for i in range(0, N):
        D = D + np.exp2(i)*B[N - i - 1, :, :].astype(np.double)

    D = D + 1

    return D


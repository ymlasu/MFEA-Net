import numpy as np

def project_input(input_array):
    '''
    Helps to project an element format binary image to a node format image.
    
    For a numpy 2d binary array input. The element 1 can project to output as 
    [[1,1],[1,1]], while 0 can project to output as [[0,0],[0,0]]. 
    The output of nearby input element will have boolean operation in common edge. '''
    rows, cols = input_array.shape
    output_shape = (rows + 1, cols + 1)
    output = np.zeros(output_shape, dtype=int)

    mask = (input_array == 1)
    output[:rows, :cols] = mask
    output[1:, 1:] = np.logical_or(output[1:, 1:], mask)
    output[1:, :cols] = np.logical_or(output[1:, :cols], mask)
    output[:rows, 1:] = np.logical_or(output[:rows, 1:], mask)

    return output

def rotate_image(image_array, angle):
    if angle == 90:
        return np.flipud(np.transpose(image_array))
    elif angle == 180:
        return np.flipud(np.fliplr(image_array))
    elif angle == 270:
        return np.fliplr(np.transpose(image_array))
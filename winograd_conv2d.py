import torch

'''
winograd_conv2d
input size: 3-dim Tensor
filter: 4-dim Tensor
'''
def winograd_conv2d(input, filters):
    H = input.shape[2]
    W = input.shape[3]
    ch = input.shape[1]     # in_channels
    n = filters.shape[0]    # count of filters
    output = torch.zeros(input.shape[0],n,H-2, W-2)
    input_size = input.shape[2]
    for i in range( input.shape[0] ):
        for row in range(0, H+1, 2):        ## stride = 2
            for col in range(0, W+1, 2):
                for ti in range(ch):
                    for to in range(n):
                        tile_y = torch.randn(2, 2)  # stride為2
                        if( row + 3 < input_size and col + 3 < input_size ):

                            tile_z = get_tile(input[i], row, col, ti) # ti 為channel
                            filter = filters[to][ti]    # which filter and its channel
                            tile_y = winograd(tile_z, filter)
                            # output中的row, col會剛好是tile_y要放的起點
                            merge_output(output[i], tile_y, row, col, to)  # to是第幾個output (第0維)

    return output

def winograd(tile_z, filter):
    '''
    tile_z size : 4x4
    filter size : 3x3
    tile_y size : 2x2
    '''
    d0 = torch.zeros(2, 3) ; d1 = torch.zeros(2, 3) ; d2 = torch.zeros(2, 3) ; d3 = torch.zeros(2, 3)
    g0 = torch.zeros(1, 3) ; g1 = torch.zeros(1, 3) ; g2 = torch.zeros(1, 3)
    result = torch.zeros(2, 2)

    d0[0] = tile_z[0][0:3] ; d0[1] = tile_z[0][1:4]
    d1[0] = tile_z[1][0:3] ; d1[1] = tile_z[1][1:4]
    d2[0] = tile_z[2][0:3] ; d2[1] = tile_z[2][1:4]
    d3[0] = tile_z[3][0:3] ; d3[1] = tile_z[3][1:4]

    g0 = filter[0] ; g1 = filter[1] ; g2 = filter[2]

    m1 = torch.matmul((d0-d2), g0)
    m2 = torch.matmul( (d1+d2), (g0+g1+g2)/2 )
    m3 = torch.matmul( (d2-d1), (g0-g1+g2)/2 )
    m4 = torch.matmul( (d1-d3), g2 )

    result[0][0] = (m1 + m2 + m3)[0]
    result[0][1] = (m1 + m2 + m3)[1]
    result[1][0] = (m2 - m3 - m4)[0]
    result[1][1] = (m2 - m3 - m4)[1]

    #print(result)

    return result

def get_tile(input, row, col, channel):
    tile = torch.zeros(4, 4)

    tile[0] = input[channel][row][col:col+4]
    tile[1] = input[channel][row+1][col:col+4]
    tile[2] = input[channel][row+2][col:col+4]
    tile[3] = input[channel][row+3][col:col+4]

    return tile

def merge_output(output, tile_y, row, col, output_channel):
    '''
    output : 4-dim
    '''
    output[output_channel][row][col] = tile_y[0][0]
    output[output_channel][row][col+1] = tile_y[0][1]
    output[output_channel][row+1][col] = tile_y[1][0]
    output[output_channel][row+1][col+1] = tile_y[1][1]





input = torch.randn(5, 3, 32, 32)
filters = torch.randn(8, 3, 3, 3)
output = winograd_conv2d(input, filters)
print(output.shape)
#print(output)

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)
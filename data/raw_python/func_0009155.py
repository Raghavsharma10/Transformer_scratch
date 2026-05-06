def _void_array_to_list(restuple, _func, _args):
    """ Convert the FFI result to Python data structures """
    shape = (restuple.e.len, 1)
    array_size = np.prod(shape)
    mem_size = 8 * array_size

    array_str_e = string_at(restuple.e.data, mem_size)
    array_str_n = string_at(restuple.n.data, mem_size)

    ls_e = np.frombuffer(array_str_e, float, array_size).tolist()
    ls_n = np.frombuffer(array_str_n, float, array_size).tolist()

    return ls_e, ls_n
def binOp(op, indx, amap, bmap, fill_vec):
    '''
    Combines the values from two map objects using the indx values
    using the op operator. In situations where there is a missing value
    it will use the callable function handle_missing
    '''
    def op_or_missing(id):
        va = amap.get(id, None)
        vb = bmap.get(id, None)
        if va is None or vb is None:
            # This should create as many elements as the number of columns!?
            result = fill_vec
        else:
            try:
                result = op(va, vb)
            except Exception:
                result = None
            if result is None:
                result = fill_vec
            return result
    seq_arys = map(op_or_missing, indx)
    data = np.vstack(seq_arys)
    return data
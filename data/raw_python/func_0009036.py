def create_table(group, name, dtype, **attributes):
    """Create a new array dataset under group with compound datatype and maxshape=(None,)"""
    dset = group.create_dataset(
        name, shape=(0,), dtype=dtype, maxshape=(None,))
    set_attributes(dset, **attributes)
    return dset
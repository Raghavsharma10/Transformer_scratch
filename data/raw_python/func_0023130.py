def cfset_to_set(cfset):
    """Convert CFSet to python set."""
    count = cf.CFSetGetCount(cfset)
    buffer = (c_void_p * count)()
    cf.CFSetGetValues(cfset, byref(buffer))
    return set([cftype_to_value(c_void_p(buffer[i])) for i in range(count)])
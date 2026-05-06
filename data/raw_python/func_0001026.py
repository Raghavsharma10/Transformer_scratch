def compat_serializer_attr(serializer, obj):
    """
    Required only for DRF 3.1, which does not make dynamically added attribute available in obj in serializer.
    This is a quick solution but works without breajing anything.
    """
    if DRFVLIST[0] == 3 and DRFVLIST[1] == 1:
        for i in serializer.instance:
            if i.id == obj.id:
                return i
    else:
        return obj
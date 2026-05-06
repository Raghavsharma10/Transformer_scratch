def set_attributes(node, overwrite=True, **attributes):
    """Set multiple attributes on node.

    If overwrite is False, and the attribute already exists, does nothing. If
    the value for a key is None, the attribute is deleted.

    """
    aset = node.attrs
    for k, v in attributes.items():
        if not overwrite and k in aset:
            pass
        elif v is None:
            if k in aset:
                del aset[k]
        else:
            aset[k] = v
def create_entry(group, name, timestamp, **attributes):
    """Create a new ARF entry under group, setting required attributes.

    An entry is an abstract collection of data which all refer to the same time
    frame. Data can include physiological recordings, sound recordings, and
    derived data such as spike times and labels. See add_data() for information
    on how data are stored.

    name -- the name of the new entry. any valid python string.

    timestamp -- timestamp of entry (datetime object, or seconds since
               January 1, 1970). Can be an integer, a float, or a tuple
               of integers (seconds, microsceconds)

    Additional keyword arguments are set as attributes on created entry.

    Returns: newly created entry object

    """
    # create group using low-level interface to store creation order
    from h5py import h5p, h5g, _hl
    try:
        gcpl = h5p.create(h5p.GROUP_CREATE)
        gcpl.set_link_creation_order(
            h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    except AttributeError:
        grp = group.create_group(name)
    else:
        name, lcpl = group._e(name, lcpl=True)
        grp = _hl.group.Group(h5g.create(group.id, name, lcpl=lcpl, gcpl=gcpl))
    set_uuid(grp, attributes.pop("uuid", None))
    set_attributes(grp,
                   timestamp=convert_timestamp(timestamp),
                   **attributes)
    return grp
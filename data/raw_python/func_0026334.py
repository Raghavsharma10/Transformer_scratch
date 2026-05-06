def fix_attr_encoding(ds):
    """ This is a temporary hot-fix to handle the way metadata is encoded
    when we read data directly from bpch files. It removes the 'scale_factor'
    and 'units' attributes we encode with the data we ingest, converts the
    'hydrocarbon' and 'chemical' attribute to a binary integer instead of a
    boolean, and removes the 'units' attribute from the "time" dimension since
    that too is implicitly encoded.

    In future versions of this library, when upstream issues in decoding
    data wrapped in dask arrays is fixed, this won't be necessary and will be
    removed.

    """

    def _maybe_del_attr(da, attr):
        """ Possibly delete an attribute on a DataArray if it's present """
        if attr in da.attrs:
            del da.attrs[attr]
        return da

    def _maybe_decode_attr(da, attr):
        # TODO: Fix this so that bools get written as attributes just fine
        """ Possibly coerce an attribute on a DataArray to an easier type
        to write to disk. """
        # bool -> int
        if (attr in da.attrs) and (type(da.attrs[attr] == bool)):
            da.attrs[attr] = int(da.attrs[attr])
        return da

    for v in ds.data_vars:
        da = ds[v]
        da = _maybe_del_attr(da, 'scale_factor')
        da = _maybe_del_attr(da, 'units')
        da = _maybe_decode_attr(da, 'hydrocarbon')
        da = _maybe_decode_attr(da, 'chemical')
    # Also delete attributes on time.
    if hasattr(ds, 'time'):
        times = ds.time
        times = _maybe_del_attr(times, 'units')

    return ds
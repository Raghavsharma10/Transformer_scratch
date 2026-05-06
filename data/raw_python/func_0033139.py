def metrique_object(_oid, _id=None, _hash=None, _start=None, _end=None,
                    _e=None, _v=None, id=None, __v__=None, **kwargs):
    '''
    Function which takes a dictionary (Mapping) object as input
    and returns return back a metrique object.

    Special meta property are added to each object::
        _oid: ...
        _start: ...
        ...
        FIXME
    '''
    # NOTE: we completely ignore incoming 'id' keys!
    # id is RESERVED and ALWAYS expected to be 'autoincrement'
    # upon insertion into DB (though, its optional, depending
    # on backend storage behaivor).
    if id:
        warnings.warn('non-null "id" keys detected, ignoring them!')

    _e = dict(_e or {})  # expecting a dict with copy() atr
    _v = int(_v or 0)

    if not isinstance(_start, float):
        _start = dt2ts(_start) if _start else utcnow(as_datetime=False)
    assert _start is not None, "_start (%s) must be set!" % _start

    if not isinstance(_end, float):
        _end = dt2ts(_end) if _end else None

    _err_msg = "_end(%s) must be >= _start(%s) or None!" % (_end, _start)
    assert _end is None or bool(_end >= _start), _err_msg

    # these meta fields are used to generate unique object _hash
    kwargs['_oid'] = _oid
    kwargs['_v'] = _v
    kwargs['_id'] = gen_id(_oid, _start, _end)  # ignore passed in _id
    # generate unique, consistent object _hash based on 'frozen' obj contents
    # FIXME: make _hash == None valid
    #kwargs['_hash'] = jsonhash(kwargs) if _hash else None
    kwargs['_hash'] = jsonhash(kwargs)

    # add some additional non-hashable meta data
    kwargs['_start'] = _start
    kwargs['_end'] = _end
    kwargs['__v__'] = __v__ or __version__
    kwargs['_e'] = _e
    return kwargs
def autodoc_tuple2doc(module):
    """Include tuples as `CLASSES` of `ControlParameters` and `RUN_METHODS`
    of `Models` into the respective docstring."""
    modulename = module.__name__
    for membername, member in inspect.getmembers(module):
        for tuplename, descr in _name2descr.items():
            tuple_ = getattr(member, tuplename, None)
            if tuple_:
                logstring = f'{modulename}.{membername}.{tuplename}'
                if logstring not in _loggedtuples:
                    _loggedtuples.add(logstring)
                    lst = [f'\n\n\n    {descr}:']
                    if tuplename == 'CLASSES':
                        type_ = 'func'
                    else:
                        type_ = 'class'
                    for cls in tuple_:
                        lst.append(
                            f'      * '
                            f':{type_}:`{cls.__module__}.{cls.__name__}`'
                            f' {objecttools.description(cls)}')
                    doc = getattr(member, '__doc__')
                    if doc is None:
                        doc = ''
                    member.__doc__ = doc + '\n'.join(l for l in lst)
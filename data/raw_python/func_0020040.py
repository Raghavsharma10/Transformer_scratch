def dict_flat_generator(value, attname=None, splitter=JSPLITTER,
                        dumps=None, prefix=None, error=ValueError,
                        recursive=True):
    '''Convert a nested dictionary into a flat dictionary representation'''
    if not isinstance(value, dict) or not recursive:
        if not prefix:
            raise error('Cannot assign a non dictionary to a JSON field')
        else:
            name = '%s%s%s' % (attname, splitter,
                               prefix) if attname else prefix
            yield name, dumps(value) if dumps else value
    else:
        # loop over dictionary
        for field in value:
            val = value[field]
            key = prefix
            if field:
                key = '%s%s%s' % (prefix, splitter,
                                  field) if prefix else field
            for k, v2 in dict_flat_generator(val, attname, splitter, dumps,
                                             key, error, field):
                yield k, v2
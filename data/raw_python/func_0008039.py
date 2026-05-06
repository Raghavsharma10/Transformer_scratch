def build_interfaces_by_method(interfaces):
    """
    Create new dictionary from INTERFACES hashed by method then
    the endpoints name. For use when using the disqusapi by the
    method interface instead of the endpoint interface. For
    instance:

    'blacklists': {
        'add': {
            'formats': ['json', 'jsonp'],
            'method': 'POST',
            'required': ['forum']
        }
    }

    is translated to:

    'POST': {
        'blacklists.add': {
            'formats': ['json', 'jsonp'],
            'method': 'POST',
            'required': ['forum']
    }
    """
    def traverse(block, parts):
        try:
            method = block['method'].lower()
        except KeyError:
            for k, v in compat.iteritems(block):
                traverse(v, parts + [k])
        else:
            path = '.'.join(parts)
            try:
                methods[method]
            except KeyError:
                methods[method] = {}
            methods[method][path] = block
    methods = {}
    for key, val in compat.iteritems(interfaces):
        traverse(val, [key])
    return methods
def parse_module(module):
    '''Parse a module's attributes and generate a markdown document.'''
    attributes = [
        (name, type_)
        for (name, type_) in getmembers(module)
        if (isclass(type_) or isfunction(type_))
        and type_.__module__ == module.__name__
        and not type_.__name__.startswith('_')
    ]

    attribute_docs = ['## {0}'.format(module.__name__), '']

    if module.__doc__:
        docstring, _ = _parse_docstring(module.__doc__)
        attribute_docs.append(docstring)

    if hasattr(module, '__all__'):
        for name in module.__all__:
            link = '+ [{0}](./{0}.md)'.format(name)
            attribute_docs.append(link)

    for (name, type_) in attributes:
        if isfunction(type_):
            attribute_docs.append(_parse_function(module, name, type_))
        else:
            attribute_docs.append(_parse_class(module, name, type_))

    return u'{0}\n'.format(
        u'\n'.join(attribute_docs).strip()
    )
def dump_cython(content, name, ofile):
    '''Generates a cython pxi file from the output of :func:`parse_header`.
    '''
    with open(ofile, 'wb') as fh:
        fh.write('cdef extern from "{}":\n'.format(name))
        for item in content:
            if isinstance(item, FunctionSpec):
                code = format_function(item)
            elif isinstance(item, StructSpec):
                code = format_struct(item)
            elif isinstance(item, EnumSpec):
                code = format_enum(item)
            elif isinstance(item, TypeDef):
                code = format_typedef(item)
            else:
                fh.write('>>>>>>>>\n{}\n<<<<<<<<'.format(item))
                code = []

            fh.write('{}\n\n'.format('\n'.join(['{}{}'.format(tab, c)
                                                for c in code])))
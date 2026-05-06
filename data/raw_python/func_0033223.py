async def convert_local(path, to_type):
    '''
    Given an absolute path to a local file, convert to a given to_type
    '''
    # Now find path between types
    typed_foreign_res = TypedLocalResource(path)
    original_ts = typed_foreign_res.typestring
    conversion_path = singletons.converter_graph.find_path(
        original_ts, to_type)
    # print('Conversion path: ', conversion_path)

    # Loop through each step in graph path and convert
    for is_first, is_last, path_step in first_last_iterator(conversion_path):
        converter_class, from_ts, to_ts = path_step
        converter = converter_class()
        in_resource = TypedLocalResource(path, from_ts)
        if is_first:  # Ensure first resource is just the source one
            in_resource = typed_foreign_res
        out_resource = TypedLocalResource(path, to_ts)

        if is_last:
            out_resource = TypedPathedLocalResource(path, to_ts)
        await converter.convert(in_resource, out_resource)
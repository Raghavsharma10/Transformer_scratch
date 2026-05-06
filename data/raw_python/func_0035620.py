def write_sourcemap(
        mappings, sources, names, output_stream, sourcemap_stream,
        normalize_paths=True, source_mapping_url=NotImplemented):
    """
    Write out the mappings, sources and names (generally produced by
    the write function) to the provided sourcemap_stream, and write the
    sourceMappingURL to the output_stream.

    Arguments

    mappings, sources, names
        These should be values produced by write function from this
        module.
    output_stream
        The original stream object that was written to; its name will
        be used for the file target and if sourceMappingURL is resolved,
        it will be writtened to this stream also as a comment.
    sourcemap_stream
        If one is provided, the sourcemap will be written out to it.

        If it is the same stream as the output_stream, the source map
        will be written as an encoded 'data:application/json;base64'
        url to the sourceMappingURL comment.  Note that an appropriate
        encoding must be available as an attribute by the output_stream
        object so that the correct character set will be used for the
        base64 encoded JSON serialized string.
    normalize_paths
        If set to True, absolute paths found will be turned into
        relative paths with relation from the stream being written
        to, and the path separator used will become a '/' (forward
        slash).
    source_mapping_url
        If an explicit value is set, this will be written as the
        sourceMappingURL into the output_stream.  Note that the path
        normalization will NOT use this value, so if paths have been
        manually provided, ensure that normalize_paths is set to False
        if the behavior is unwanted.
    """

    encode_sourcemap_args, output_js_map = verify_write_sourcemap_args(
        mappings, sources, names, output_stream, sourcemap_stream,
        normalize_paths
    )

    encoded_sourcemap = json.dumps(
        encode_sourcemap(*encode_sourcemap_args),
        sort_keys=True, ensure_ascii=False,
    )

    if sourcemap_stream is output_stream:
        # encoding will be missing if using StringIO; fall back to
        # default_encoding
        encoding = getattr(output_stream, 'encoding', None) or default_encoding
        output_stream.writelines([
            '\n//# sourceMappingURL=data:application/json;base64;charset=',
            encoding, ',', base64.b64encode(
                encoded_sourcemap.encode(encoding)).decode('ascii'),
        ])
    else:
        if source_mapping_url is not None:
            output_stream.writelines(['\n//# sourceMappingURL=', (
                output_js_map if source_mapping_url is NotImplemented
                else source_mapping_url
            ), '\n'])

        sourcemap_stream.write(encoded_sourcemap)
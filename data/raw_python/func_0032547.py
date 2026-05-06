def write_tsv(output_stream, *tup, **kwargs):
    """
    Write argument list in `tup` out as a tab-separeated row to the stream.
    """
    encoding = kwargs.get('encoding') or 'utf-8'
    value = '\t'.join([s for s in tup]) + '\n'
    output_stream.write(value.encode(encoding))
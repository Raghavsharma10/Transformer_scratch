def get_single_outfilename(args):
    """Use first possible entry in query as filename."""
    for arg in args['query']:
        if arg in args['files']:
            return ('.'.join(arg.split('.')[:-1])).lower()
        for url in args['urls']:
            if arg.strip('/') in url:
                domain = get_domain(url)
                return get_outfilename(url, domain)
    sys.stderr.write('Failed to construct a single out filename.\n')
    return ''
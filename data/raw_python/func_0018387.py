def peep_port(paths):
    """Convert a peep requirements file to one compatble with pip-8 hashing.

    Loses comments and tromps on URLs, so the result will need a little manual
    massaging, but the hard part--the hash conversion--is done for you.

    """
    if not paths:
        print('Please specify one or more requirements files so I have '
              'something to port.\n')
        return COMMAND_LINE_ERROR

    comes_from = None
    for req in chain.from_iterable(
            _parse_requirements(path, package_finder(argv)) for path in paths):
        req_path, req_line = path_and_line(req)
        hashes = [hexlify(urlsafe_b64decode((hash + '=').encode('ascii'))).decode('ascii')
                  for hash in hashes_above(req_path, req_line)]
        if req_path != comes_from:
            print()
            print('# from %s' % req_path)
            print()
            comes_from = req_path

        if not hashes:
            print(req.req)
        else:
            print('%s' % (req.link if getattr(req, 'link', None) else req.req), end='')
            for hash in hashes:
                print(' \\')
                print('    --hash=sha256:%s' % hash, end='')
            print()
def init(pandocversion=None, doc=None):
    """Sets or determines the pandoc version.  This must be called.

    The pandoc version is needed for multi-version support.
    See: https://github.com/jgm/pandoc/issues/2640

    Returns the pandoc version."""

    # This requires some care because we can't be sure that a call to 'pandoc'
    # will work.  It could be 'pandoc-1.17.0.2' or some other name.  Try
    # checking the parent process first, and only make a call to 'pandoc' as
    # a last resort.

    global _PANDOCVERSION  # pylint: disable=global-statement

    pattern = re.compile(r'^[1-2]\.[0-9]+(?:\.[0-9]+)?(?:\.[0-9]+)?$')

    if 'PANDOC_VERSION' in os.environ:  # Available for pandoc >= 1.19.1
        pandocversion = str(os.environ['PANDOC_VERSION'])

    if not pandocversion is None:
        # Test the result and if it is OK then store it in _PANDOCVERSION
        if pattern.match(pandocversion):
            _PANDOCVERSION = pandocversion
            return _PANDOCVERSION
        else:
            msg = 'Cannot understand pandocversion=%s'%pandocversion
            raise RuntimeError(msg)

    if not doc is None:
        if 'pandoc-api-version' in doc:
            # This could be either 1.18 or 1.19; there is no way to
            # distinguish them (but there isn't a use case in pandoc-fignos
            # and friends where it matters)
            _PANDOCVERSION = '1.18'
            return _PANDOCVERSION

    # Get the command
    try:  # Get the path for the parent process
        if os.name == 'nt':
            # psutil appears to work differently for windows
            command = psutil.Process(os.getpid()).parent().parent().exe()
        else:
            command = psutil.Process(os.getpid()).parent().exe()
        if not os.path.basename(command).startswith('pandoc'):
            raise RuntimeError('pandoc not found')
    except:  # pylint: disable=bare-except
        # Call whatever pandoc is available and hope for the best
        command = 'pandoc'

    # Make the call
    try:
        # Get the version number and confirm it conforms to expectations
        output = subprocess.check_output([command, '-v'])
        line = output.decode('utf-8').split('\n')[0]
        pandocversion = line.split(' ')[-1].strip()
    except: # pylint: disable=bare-except
        pandocversion = ''

    # Test the result and if it is OK then store it in _PANDOCVERSION
    if pattern.match(pandocversion):
        _PANDOCVERSION = pandocversion

    if _PANDOCVERSION is None:
        msg = """Cannot determine pandoc version.  Please file an issue at
              https://github.com/tomduck/pandocfiltering/issues"""
        raise RuntimeError(textwrap.dedent(msg))

    return _PANDOCVERSION
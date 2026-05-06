def package_finder(argv):
    """Return a PackageFinder respecting command-line options.

    :arg argv: Everything after the subcommand

    """
    # We instantiate an InstallCommand and then use some of its private
    # machinery--its arg parser--for our own purposes, like a virus. This
    # approach is portable across many pip versions, where more fine-grained
    # ones are not. Ignoring options that don't exist on the parser (for
    # instance, --use-wheel) gives us a straightforward method of backward
    # compatibility.
    try:
        command = InstallCommand()
    except TypeError:
        # This is likely pip 1.3.0's "__init__() takes exactly 2 arguments (1
        # given)" error. In that version, InstallCommand takes a top=level
        # parser passed in from outside.
        from pip.baseparser import create_main_parser
        command = InstallCommand(create_main_parser())
    # The downside is that it essentially ruins the InstallCommand class for
    # further use. Calling out to pip.main() within the same interpreter, for
    # example, would result in arguments parsed this time turning up there.
    # Thus, we deepcopy the arg parser so we don't trash its singletons. Of
    # course, deepcopy doesn't work on these objects, because they contain
    # uncopyable regex patterns, so we pickle and unpickle instead. Fun!
    options, _ = loads(dumps(command.parser)).parse_args(argv)

    # Carry over PackageFinder kwargs that have [about] the same names as
    # options attr names:
    possible_options = [
        'find_links',
        FORMAT_CONTROL_ARG,
        ('allow_all_prereleases', 'pre'),
        'process_dependency_links'
    ]
    kwargs = {}
    for option in possible_options:
        kw, attr = option if isinstance(option, tuple) else (option, option)
        value = getattr(options, attr, MARKER)
        if value is not MARKER:
            kwargs[kw] = value

    # Figure out index_urls:
    index_urls = [options.index_url] + options.extra_index_urls
    if options.no_index:
        index_urls = []
    index_urls += getattr(options, 'mirrors', [])

    # If pip is new enough to have a PipSession, initialize one, since
    # PackageFinder requires it:
    if hasattr(command, '_build_session'):
        kwargs['session'] = command._build_session(options)

    return PackageFinder(index_urls=index_urls, **kwargs)
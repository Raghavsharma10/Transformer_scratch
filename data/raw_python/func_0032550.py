def parse_cmdln_opts(parser, cmdln_args):
    """Rather than have this all clutter main(), let's split this out.
    Clean arch decision: rather than parsing sys.argv directly, pass
    sys.argv[1:] to this function (or any iterable for testing.)
    """
    parser.set_defaults(
        hosts=[],
        cert=None,
        log_level=logging.INFO,
        output_dir=None,
        output_file=None,
        formats=[],
        includes=[],
        excludes=[],
        nsscmd=None,
        tokenfile=None,
        noncefile=None,
        cachedir=None,
    )

    parser.add_option(
        "-H", "--host", dest="hosts", action="append", help="format[:format]:hostname[:port]")
    parser.add_option("-c", "--server-cert", dest="cert")
    parser.add_option("-t", "--token-file", dest="tokenfile",
                      help="file where token is stored")
    parser.add_option("-n", "--nonce-file", dest="noncefile",
                      help="file where nonce is stored")
    parser.add_option("-d", "--output-dir", dest="output_dir",
                      help="output directory; if not set then files are "
                      "replaced with signed copies")
    parser.add_option("-o", "--output-file", dest="output_file",
                      help="output file; if not set then files are replaced with signed "
                      "copies. This can only be used when signing a single file")
    parser.add_option("-f", "--formats", dest="formats", action="append",
                      help="signing formats (one or more of %s)" % ", ".join(ALLOWED_FORMATS))
    parser.add_option("-q", "--quiet", dest="log_level", action="store_const",
                      const=logging.WARN)
    parser.add_option(
        "-v", "--verbose", dest="log_level", action="store_const",
        const=logging.DEBUG)
    parser.add_option("-i", "--include", dest="includes", action="append",
                      help="add to include patterns")
    parser.add_option("-x", "--exclude", dest="excludes", action="append",
                      help="add to exclude patterns")
    parser.add_option("--nsscmd", dest="nsscmd",
                      help="command to re-sign nss libraries, if required")
    parser.add_option("--cachedir", dest="cachedir",
                      help="local cache directory")
    # TODO: Concurrency?
    # TODO: Different certs per server?

    options, args = parser.parse_args(cmdln_args)

    if not options.hosts:
        parser.error("at least one host is required")

    if not options.cert:
        parser.error("certificate is required")

    if not os.path.exists(options.cert):
        parser.error("certificate not found")

    if not options.tokenfile:
        parser.error("token file is required")

    if not options.noncefile:
        parser.error("nonce file is required")

    # Covert nsscmd to win32 path if required
    if sys.platform == 'win32' and options.nsscmd:
        nsscmd = options.nsscmd.strip()
        if nsscmd.startswith("/"):
            drive = nsscmd[1]
            options.nsscmd = "%s:%s" % (drive, nsscmd[2:])

    # Handle format
    formats = []
    for fmt in options.formats:
        if "," in fmt:
            for fmt in fmt.split(","):
                if fmt not in ALLOWED_FORMATS:
                    parser.error("invalid format: %s" % fmt)
                formats.append(fmt)
        elif fmt not in ALLOWED_FORMATS:
            parser.error("invalid format: %s" % fmt)
        else:
            formats.append(fmt)

    # bug 1382882, 1164456
    # Widevine and GPG signing must happen last because they will be invalid if
    # done prior to any format that modifies the file in-place.
    for fmt in ("widevine", "widevine_blessed", "gpg"):
        if fmt in formats:
            formats.remove(fmt)
            formats.append(fmt)

    if options.output_file and (len(args) > 1 or os.path.isdir(args[0])):
        parser.error(
            "-o / --output-file can only be used when signing a single file")

    if options.output_dir:
        if os.path.exists(options.output_dir):
            if not os.path.isdir(options.output_dir):
                parser.error(
                    "output_dir (%s) must be a directory", options.output_dir)
        else:
            os.makedirs(options.output_dir)

    if not options.includes:
        # Do everything!
        options.includes.append("*")

    if not formats:
        parser.error("no formats specified")

    options.formats = formats

    format_urls = defaultdict(list)
    for h in options.hosts:
        # The last two parts of a host is the actual hostname:port. Any parts
        # before that are formats - there could be 0..n formats so this is
        # tricky to split.
        parts = h.split(":")
        h = parts[-2:]
        fmts = parts[:-2]
        # If no formats are specified, the host is assumed to support all of them.
        if not fmts:
            fmts = formats

        for f in fmts:
            format_urls[f].append("https://%s" % ":".join(h))
    options.format_urls = format_urls

    missing_fmt_hosts = set(formats) - set(format_urls.keys())
    if missing_fmt_hosts:
        parser.error("no hosts capable of signing formats: %s" % " ".join(missing_fmt_hosts))

    return options, args
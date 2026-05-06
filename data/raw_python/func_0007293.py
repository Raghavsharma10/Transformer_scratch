def launcher():
    """Launch it."""
    parser = OptionParser()
    parser.add_option(
        '-f',
        '--file',
        dest='filename',
        default='agents.csv',
        help='snmposter configuration file'
    )
    options, args = parser.parse_args()

    factory = SNMPosterFactory()

    snmpd_status = subprocess.Popen(
        ["service", "snmpd", "status"],
        stdout=subprocess.PIPE
    ).communicate()[0]

    if "is running" in snmpd_status:
        message = "snmd service is running. Please stop it and try again."
        print >> sys.stderr, message
        sys.exit(1)

    try:
        factory.configure(options.filename)
    except IOError:
        print >> sys.stderr, "Error opening %s." % options.filename
        sys.exit(1)

    factory.start()
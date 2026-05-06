def main(argv=None):
    """Run Nutch command using REST API."""
    global Verbose, Mock
    if argv is None:
        argv = sys.argv

    if len(argv) < 5: die('Bad args')
    try:
        opts, argv = getopt.getopt(argv[1:], 'hs:p:mv',
          ['help', 'server=', 'port=', 'mock', 'verbose'])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        die()

    serverEndpoint = DefaultServerEndpoint
    # TODO: Fix this
    for opt, val in opts:
        if opt   in ('-h', '--help'):    echo2(USAGE); sys.exit()
        elif opt in ('-s', '--server'):  serverEndpoint = val
        elif opt in ('-p', '--port'):    serverEndpoint = 'http://localhost:%s' % val
        elif opt in ('-m', '--mock'):    Mock = 1
        elif opt in ('-v', '--verbose'): Verbose = 1
        else: die(USAGE)

    cmd = argv[0]
    crawlId = argv[1]
    confId = argv[2]
    urlDir = argv[3]
    args = {}
    if len(argv) > 4: args = eval(argv[4])

    nt = Nutch(crawlId, confId, serverEndpoint, urlDir)
    nt.Jobs().create(cmd, **args)
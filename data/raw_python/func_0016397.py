def main():
    """parse commandline arguments and print result"""

    #
    # setup command line parsing a la argpase
    #
    parser = argparse.ArgumentParser()

    # positional args
    parser.add_argument('uri', metavar='URI', nargs='?', default='/',
                        help='[owserver:]//hostname:port/path')

    #
    # parse command line args
    #
    args = parser.parse_args()

    #
    # parse args.uri and substitute defaults
    #
    urlc = urlsplit(args.uri, scheme='owserver', allow_fragments=False)
    assert urlc.fragment == ''
    if urlc.scheme != 'owserver':
        parser.error("Invalid URI scheme '{0}:'".format(urlc.scheme))
    if urlc.query:
        parser.error("Invalid URI, query component '?{0}' not allowed"
                     .format(urlc.query))
    try:
        host = urlc.hostname or 'localhost'
        port = urlc.port or 4304
    except ValueError as error:
        parser.error("Invalid URI: invalid net location '//{0}/'"
                     .format(urlc.netloc))

    #
    # create owserver proxy object
    #
    try:
        owproxy = protocol.proxy(host, port, persistent=True)
    except protocol.ConnError as error:
        print("Unable to open connection to '{0}:{1}'\n{2}"
              .format(host, port, error), file=sys.stderr)
        sys.exit(1)
    except protocol.ProtocolError as error:
        print("Protocol error, '{0}:{1}' not an owserver?\n{2}"
              .format(host, port, error), file=sys.stderr)
        sys.exit(1)

    stress(owproxy, urlc.path)
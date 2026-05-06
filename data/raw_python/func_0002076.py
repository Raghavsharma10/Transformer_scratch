def console_create():
    """
    Command line tool (``koordinates-create-token``) used to create an API token.
    """
    import argparse
    import getpass
    import re
    import sys
    import requests
    from six.moves import input
    from koordinates.client import Client

    parser = argparse.ArgumentParser(description="Command line tool to create a Koordinates API Token.")
    parser.add_argument('site', help="Domain (eg. labs.koordinates.com) for the Koordinates site.", metavar='SITE')
    parser.add_argument('email', help="User account email address", metavar='EMAIL')
    parser.add_argument('name', help="Description for the key", metavar='NAME')
    parser.add_argument('--scopes', help="Scopes for the new API token", nargs='+', metavar="SCOPE")
    parser.add_argument('--referrers', help="Restrict the request referrers for the token. You can use * as a wildcard, eg. *.example.com", nargs='+', metavar='HOST')
    parser.add_argument('--expires', help="Expiry time in ISO 8601 (YYYY-MM-DD) format", metavar="DATE")
    args = parser.parse_args()

    # check we have a valid-ish looking domain name
    if not re.match(r'(?=^.{4,253}$)(^((?!-)[a-zA-Z0-9-]{1,63}(?<!-)\.)+[a-zA-Z]{2,63}$)', args.site):
        parser.error("'%s' doesn't look like a valid domain name." % args.site)

    # check we have a valid-ish looking email address
    if not re.match(r'[^@]+@[^@]+\.[^@]+$', args.email):
        parser.error("'%s' doesn't look like a valid email address." % args.email)

    password = getpass.getpass('Account password for %s: ' % args.email)
    if not password:
        parser.error("Empty password.")

    expires_at = make_date(args.expires)

    print("\nNew API Token Parameters:")
    print("  Koordinates Site: %s" % args.site)
    print("  User email address: %s" % args.email)
    print("  Name: %s" % args.name)
    print("  Scopes: %s" % ' '.join(args.scopes or ['(default)']))
    print("  Referrer restrictions: %s" % ' '.join(args.referrers or ['(none)']))
    print("  Expires: %s" % (expires_at or '(never)'))
    if input("Continue? [Yn] ").lower() == 'n':
        sys.exit(1)

    token = Token(name=args.name)
    if args.scopes:
        token.scopes = args.scopes
    if args.referrers:
        token.referrers = args.referrers
    if expires_at:
        token.expires_at = expires_at

    print("\nRequesting token...")
    # need a dummy token here for initialisation
    client = Client(host=args.site, token='-dummy-')
    try:
        token = client.tokens.create(token, args.email, password)
    except requests.HTTPError as e:
        print("%s: %s" % (e, e.response.text))

        # Helpful tips for specific errors
        if e.response.status_code == 401:
            print("  => Check your email address, password, and site domain carefully.")

        sys.exit(1)

    print("Token created successfully.\n  Key: %s\n  Scopes: %s" % (token.key, token.scope))
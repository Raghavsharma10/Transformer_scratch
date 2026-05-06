def authorize(args):
    """
    Authorizes Coursera's OAuth2 client for using coursera.org API servers for
    a specific application
    """
    oauth2_instance = oauth2.build_oauth2(args.app, args)
    oauth2_instance.build_authorizer()
    logging.info('Application "%s" authorized!', args.app)
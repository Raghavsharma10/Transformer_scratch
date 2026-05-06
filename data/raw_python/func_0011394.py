def main(args,parser,subparser=None):
    '''this is the main entrypoint for a container based web server, with
       most of the variables coming from the environment. See the Dockerfile
       template for how this function is executed.

    '''

    # First priority to args.base
    base = args.base
    if base is None:
        base = os.environ.get('EXPFACTORY_BASE')

    # Does the base folder exist?
    if base is None:
        bot.error("You must set a base of experiments with --base" % base)
        sys.exit(1)

    if not os.path.exists(base):
        bot.error("Base folder %s does not exist." % base)
        sys.exit(1)

    # Export environment variables for the client
    experiments = args.experiments
    if experiments is None:
        experiments = " ".join(glob("%s/*" % base))

    os.environ['EXPFACTORY_EXPERIMENTS'] = experiments

    # If defined and file exists, set runtime variables
    if args.vars is not None:
        if os.path.exists(args.vars):
            os.environ['EXPFACTORY_RUNTIME_VARS'] = args.vars
            os.environ['EXPFACTORY_RUNTIME_DELIM'] = args.delim
        else:
            bot.warning('Variables file %s not found.' %args.vars)


    subid = os.environ.get('EXPFACTORY_STUDY_ID')
    if args.subid is not None:
        subid = args.subid 
        os.environ['EXPFACTORY_SUBID'] = subid

    os.environ['EXPFACTORY_RANDOM'] = str(args.disable_randomize)
    os.environ['EXPFACTORY_BASE'] = base
    
    from expfactory.server import start
    start(port=5000)
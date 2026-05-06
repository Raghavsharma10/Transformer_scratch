def static(path, no_input):
    '''Compile and collect static files into path'''
    log = logging.getLogger('webassets')
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.DEBUG)

    cmdenv = CommandLineEnvironment(assets, log)
    cmdenv.build()

    if exists(path):
        warning('{0} directory already exists and will be {1}', white(path), white('erased'))
        if not no_input and not click.confirm('Are you sure'):
            exit_with_error()
        shutil.rmtree(path)

    echo('Copying assets into {0}', white(path))
    shutil.copytree(assets.directory, path)

    success('Done')
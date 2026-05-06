def reload_(environment, opts):
    """Reload environment source and configuration

Usage:
  datacats reload [-b] [-p|--no-watch] [--syslog] [-s NAME] [--site-url=SITE_URL]
                            [-i] [--address=IP] [ENVIRONMENT [PORT]]
  datacats reload -r [-b] [--syslog] [-s NAME] [--address=IP] [--site-url=SITE_URL]
                            [-i] [ENVIRONMENT]

Options:
  --address=IP          Address to listen on (Linux-only)
  -i --interactive      Calls out to docker via the command line, allowing
                        for interactivity with the web image.
  --site-url=SITE_URL   The site_url to use in API responses. Can use Python template syntax
                        to insert the port and address (e.g. http://example.org:{port}/)
  -b --background       Don't wait for response from web server
  --no-watch            Do not automatically reload templates and .py files on change
  -p --production       Reload with apache and debug=false
  -s --site=NAME        Specify a site to reload [default: primary]
  --syslog              Log to the syslog

ENVIRONMENT may be an environment name or a path to an environment directory.
Default: '.'
"""
    if opts['--interactive']:
        # We can't wait for the server if we're tty'd
        opts['--background'] = True
    if opts['--address'] and is_boot2docker():
        raise DatacatsError('Cannot specify address on boot2docker.')
    environment.require_data()
    environment.stop_ckan()
    if opts['PORT'] or opts['--address'] or opts['--site-url']:
        if opts['PORT']:
            environment.port = int(opts['PORT'])
        if opts['--address']:
            environment.address = opts['--address']
        if opts['--site-url']:
            site_url = opts['--site-url']
            # TODO: Check it against a regex or use urlparse
            try:
                site_url = site_url.format(address=environment.address, port=environment.port)
                environment.site_url = site_url
                environment.save_site(False)
            except (KeyError, IndexError, ValueError) as e:
                raise DatacatsError('Could not parse site_url: {}'.format(e))
        environment.save()

    for container in environment.extra_containers:
        require_extra_image(EXTRA_IMAGE_MAPPING[container])

    environment.stop_supporting_containers()
    environment.start_supporting_containers()

    environment.start_ckan(
        production=opts['--production'],
        paster_reload=not opts['--no-watch'],
        log_syslog=opts['--syslog'],
        interactive=opts['--interactive'])
    write('Starting web server at {0} ...'.format(environment.web_address()))
    if opts['--background']:
        write('\n')
        return

    try:
        environment.wait_for_web_available()
    finally:
        write('\n')
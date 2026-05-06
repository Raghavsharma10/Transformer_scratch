def create(opts):
    """Create a new environment

Usage:
  datacats create [-bin] [--interactive] [-s NAME] [--address=IP] [--syslog]
                  [--ckan=CKAN_VERSION] [--no-datapusher] [--site-url SITE_URL]
                  [--no-init-db] ENVIRONMENT_DIR [PORT]

Options:
  --address=IP            Address to listen on (Linux-only)
  --ckan=CKAN_VERSION     Use CKAN version CKAN_VERSION [default: 2.4]
  -b --bare               Bare CKAN site with no example extension
  -i --image-only         Create the environment but don't start containers
  --interactive           Doesn't detach from the web container
  --no-datapusher         Don't install/enable ckanext-datapusher
  --no-init-db            Don't initialize the database. Useful for importing CKANs.
  -n --no-sysadmin        Don't prompt for an initial sysadmin user account
  -s --site=NAME          Pick a site to create [default: primary]
  --site-url SITE_URL     The site_url to use in API responses (e.g. http://example.org:{port}/)
  --syslog                Log to the syslog

ENVIRONMENT_DIR is a path for the new environment directory. The last
part of this path will be used as the environment name.
"""
    if opts['--address'] and is_boot2docker():
        raise DatacatsError('Cannot specify address on boot2docker.')
    return create_environment(
        environment_dir=opts['ENVIRONMENT_DIR'],
        port=opts['PORT'],
        create_skin=not opts['--bare'],
        start_web=not opts['--image-only'],
        create_sysadmin=not opts['--no-sysadmin'],
        site_name=opts['--site'],
        ckan_version=opts['--ckan'],
        address=opts['--address'],
        log_syslog=opts['--syslog'],
        datapusher=not opts['--no-datapusher'],
        site_url=opts['--site-url'],
        interactive=opts['--interactive'],
        init_db=not opts['--no-init-db'],
        )
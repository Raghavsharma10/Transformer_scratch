def update(args):
    """
    Run site update
    ---------------

    Run updates for site or all installed.

    ::

        usage: main.py update [-h] [-v] [-p PATH] [SITE]

        Run site update

        positional arguments:
        SITE                  Path to site or name (project.branch)

        optional arguments:
        -p PATH, --path PATH  path to makesite sites instalation dir. you can set it
                                in $makesite_home env variable.

    Examples: ::

        # Update all makesite instances on server
        $ makesite update

        # Update by project name
        makesite update intaxi

        # Update by project name
        makesite update intaxi.develop

        # Update by project path
        makesite update /var/www/intaxi/master


    """
    if args.SITE:
        site = find_site(args.SITE, path=args.path)
        return site.run_update()

    for site in gen_sites(args.path):
        site.run_update()
    return True
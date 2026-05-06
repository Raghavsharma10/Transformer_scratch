def packages(ciprcfg, env, opts, console):
    """
    List installed packages for this project
    """
    for name, source in ciprcfg.packages.items():
        console.normal('- %s' % name)

        if opts.long_details:
            console.normal('  - directory: %s' % path.join(env.package_dir, name))
            console.normal('  - source: %s' % source)
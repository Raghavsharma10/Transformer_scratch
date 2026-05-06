def uninstall(args, env, console, ciprcfg):
    """
    Remove a package
    """
    for name in args:
        package_dir = path.join(env.package_dir, name)
        if path.exists(package_dir):
            console.quiet('Removing %s...' % name)
            if path.islink(package_dir):
                os.remove(package_dir)
            else:
                shutil.rmtree(package_dir)

        ciprcfg.remove_package(name)
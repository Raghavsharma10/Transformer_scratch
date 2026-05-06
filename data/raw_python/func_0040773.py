def install(args, console, env, ciprcfg, opts):
    """
    Install a package from github and make it available for use.
    """
    if len(args) == 0:
        # Is this a cipr project?
        if ciprcfg.exists:
            # Install all the packages for this project
            console.quiet('Installing current project packages...')
            for name, source in ciprcfg.packages.items():
                if opts.upgrade:
                    app.command.run(['install', '--upgrade', source])
                else:
                    app.command.run(['install', source])
        else:
            console.error('No cipr project or package found.')
        return
    else:
        for source in args:
            package, name, version, type = _package_info(source)

            if not path.exists(env.package_dir):
                os.makedirs(env.package_dir)

            package_dir = path.join(env.package_dir, name)

            if path.exists(package_dir):
                if opts.upgrade:
                    app.command.run(['uninstall', name])
                else:
                    console.quiet('Package %s already exists. Use --upgrade to force a re-install.' % name)
                    return

            console.quiet('Installing %s...' % name)


            if type == 'git':
                tmpdir = tempfile.mkdtemp(prefix='cipr')
                clom.git.clone(package, tmpdir).shell.execute()

                if version:
                    cmd = AND(clom.cd(tmpdir), clom.git.checkout(version))
                    cmd.shell.execute()

                package_json = path.join(tmpdir, 'package.json')
                if path.exists(package_json):
                    # Looks like a cipr package, copy directly
                    shutil.move(tmpdir, package_dir)
                else:
                    # Not a cipr package, sandbox in sub-directory
                    shutil.move(tmpdir, path.join(package_dir, name))

                console.quiet('`%s` installed from git repo to `%s`' % (name, package_dir))

            elif path.exists(package):
                # Local
                os.symlink(package, package_dir)
            else:
                console.error('Package `%s` type not recognized' % package)
                return

            pkg = Package(package_dir, source)
            ciprcfg.add_package(pkg)

            if pkg.dependencies:
                console.quiet('Installing dependancies...')
                for name, require in pkg.dependencies.items():
                    if opts.upgrade:
                        app.command.run(['install', '--upgrade', require])
                    else:
                        app.command.run(['install', require])
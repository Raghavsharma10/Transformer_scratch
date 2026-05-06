def install_modules(wip):
    """Install the plugin modules"""

    def install_module(hfos_module):
        """Install a single module via setuptools"""
        try:
            setup = Popen(
                [
                    sys.executable,
                    'setup.py',
                    'develop'
                ],
                cwd='modules/' + hfos_module + "/"
            )

            setup.wait()
        except Exception as e:
            log("Problem during module installation: ", hfos_module, e,
                type(e), exc=True, lvl=error)
            return False
        return True

    # TODO: Sort module dependencies via topological sort or let pip do this in future.
    # # To get the module dependencies:
    # packages = {}
    # for provision_entrypoint in iter_entry_points(group='hfos.provisions',
    #                                               name=None):
    #     log("Found packages: ", provision_entrypoint.dist.project_name, lvl=warn)
    #
    #     _package_name = provision_entrypoint.dist.project_name
    #     _package = pkg_resources.working_set.by_key[_package_name]
    #
    #     print([str(r) for r in _package.requires()])  # retrieve deps from setup.py

    modules_production = [
        # TODO: Poor man's dependency management, as long as the modules are
        # installed from local sources and they're not available on pypi,
        # which would handle real dependency management for us:
        'navdata',

        # Now all the rest:
        'alert',
        'automat',
        'busrepeater',
        'calendar',
        'countables',
        'dash',
        # 'dev',
        'enrol',
        'mail',
        'maps',
        'nmea',
        'nodestate',
        'project',
        'webguides',
        'wiki'
    ]

    modules_wip = [
        'calc',
        'camera',
        'chat',
        'comms',
        'contacts',
        'crew',
        'equipment',
        'filemanager',
        'garden',
        'heroic',
        'ldap',
        'library',
        'logbook',
        'protocols',
        'polls',
        'mesh',
        'robot',
        'switchboard',
        'shareables',
    ]

    installables = modules_production

    if wip:
        installables.extend(modules_wip)

    success = []
    failed = []

    for installable in installables:
        log('Installing module ', installable)
        if install_module(installable):
            success.append(installable)
        else:
            failed.append(installable)

    log('Installed modules: ', success)
    if len(failed) > 0:
        log('Failed modules: ', failed)
    log('Done: Install Modules')
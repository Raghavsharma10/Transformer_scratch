def _handle_args(self, cmd, args):
        """
        We need to support deprecated behaviour for now which makes this
        quite complicated

        Current behaviour:
        - install: Installs a new server, existing server causes an error
        - install --upgrade: Installs or upgrades a server
        - install --managedb: Automatically initialise or upgrade the db

        Deprecated:
        - install --upgradedb --initdb: Replaced by install --managedb
        - install --upgradedb: upgrade the db, must exist
        - install --initdb: initialise the db
        - upgrade: Upgrades a server, must already exist
        - upgrade --upgradedb: Automatically upgrade the db

        returns:
        - Modified args object, flag to indicate new/existing/auto install
        """
        if cmd == 'install':
            if args.upgrade:
                # Current behaviour: install or upgrade
                if args.initdb or args.upgradedb:
                    raise Stop(10, (
                        'Deprecated --initdb --upgradedb flags '
                        'are incompatible with --upgrade'))
                newinstall = None
            else:
                # Current behaviour: Server must not exist
                newinstall = True

            if args.managedb:
                # Current behaviour
                if args.initdb or args.upgradedb:
                    raise Stop(10, (
                        'Deprecated --initdb --upgradedb flags '
                        'are incompatible with --managedb'))
                args.initdb = True
                args.upgradedb = True
            else:
                if args.initdb or args.upgradedb:
                    log.warn('--initdb and --upgradedb are deprecated, '
                             'use --managedb')

        elif cmd == 'upgrade':
            # Deprecated behaviour
            log.warn(
                '"omero upgrade" is deprecated, use "omego install --upgrade"')
            cmd = 'install'
            args.upgrade = True
            # Deprecated behaviour: Server must exist
            newinstall = False

        else:
            raise Exception('Unexpected command: %s' % cmd)

        return args, newinstall
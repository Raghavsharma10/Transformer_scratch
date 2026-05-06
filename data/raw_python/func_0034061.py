def handle(self, *args, **options):
        """
        Handle liquibase command parameters
        """
        database = getattr(
                settings, 'LIQUIMIGRATE_DATABASE', options['database'])

        try:
            dbsettings = databases[database]
        except KeyError:
            raise CommandError("don't know such a connection: %s" % database)

        verbosity = int(options.get('verbosity'))

        # get driver
        driver_class = (
                options.get('driver')
                or dbsettings.get('ENGINE').split('.')[-1])
        dbtag, driver, classpath = LIQUIBASE_DRIVERS.get(
                            driver_class, (None, None, None))

        classpath = options.get('classpath') or classpath

        if driver is None:
            raise CommandError(
                "unsupported db driver '%s'\n"
                "available drivers: %s" % (
                    driver_class, ' '.join(LIQUIBASE_DRIVERS.keys())))

        # command options
        changelog_file = (
                options.get('changelog_file')
                or _get_changelog_file(options['database']))
        username = options.get('username') or dbsettings.get('USER') or ''
        password = options.get('password') or dbsettings.get('PASSWORD') or ''
        url = options.get('url') or _get_url_for_db(dbtag, dbsettings)

        command = options['command']
        cmdargs = {
            'jar': LIQUIBASE_JAR,
            'changelog_file': changelog_file,
            'username': username,
            'password': password,
            'command': command,
            'driver': driver,
            'classpath': classpath,
            'url': url,
            'args': ' '.join(args),
        }

        cmdline = "java -jar %(jar)s --changeLogFile %(changelog_file)s \
--username=%(username)s --password=%(password)s \
--driver=%(driver)s --classpath=%(classpath)s --url=%(url)s \
%(command)s %(args)s" % (cmdargs)

        if verbosity > 0:
            print("changelog file: %s" % (changelog_file,))
            print("executing: %s" % (cmdline,))

        created_models = None   # we dont know it

        if emit_pre_migrate_signal and not options.get('no_signals'):
            if django_19_or_newer:
                emit_pre_migrate_signal(
                        1, options.get('interactive'), database)
            else:
                emit_pre_migrate_signal(
                    created_models, 1, options.get('interactive'), database)

        rc = os.system(cmdline)

        if rc == 0:

            try:
                if not options.get('no_signals'):
                    if emit_post_migrate_signal:
                        if django_19_or_newer:
                            emit_post_migrate_signal(
                                0, options.get('interactive'), database)
                        else:
                            emit_post_migrate_signal(
                                created_models, 0,
                                options.get('interactive'), database)
                    elif emit_post_sync_signal:
                        emit_post_sync_signal(
                                created_models, 0,
                                options.get('interactive'), database)

                if not django_19_or_newer:
                    call_command(
                        'loaddata', 'initial_data', verbosity=1,
                        database=database)
            except TypeError:
                # singledb (1.1 and older)
                emit_post_sync_signal(
                        created_models, 0, options.get('interactive'))

                call_command(
                        'loaddata', 'initial_data', verbosity=0)
        else:
            raise CommandError('Liquibase returned an error code %s' % rc)
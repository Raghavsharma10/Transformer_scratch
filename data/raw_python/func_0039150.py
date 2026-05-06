def run_outdated(cls, options):
        """Print outdated user packages."""
        latest_versions = sorted(
            cls.find_packages_latest_versions(cls.options),
            key=lambda p: p[0].project_name.lower())

        for dist, latest_version, typ in latest_versions:
            if latest_version > dist.parsed_version:
                if options.all:
                    pass
                elif options.pinned:
                    if cls.can_be_updated(dist, latest_version):
                        continue
                elif not options.pinned:
                    if not cls.can_be_updated(dist, latest_version):
                        continue
                    elif options.update:
                        print(dist.project_name if options.brief else
                              'Updating %s to Latest: %s [%s]' %
                              (cls.output_package(dist), latest_version, typ))
                        main(['install', '--upgrade'] + ([
                            '--user'
                        ] if ENABLE_USER_SITE else []) + [dist.key])
                        continue

                print(dist.project_name if options.brief else
                      '%s - Latest: %s [%s]' %
                      (cls.output_package(dist), latest_version, typ))
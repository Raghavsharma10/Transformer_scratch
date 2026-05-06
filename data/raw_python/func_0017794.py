def start_supporting_containers(sitedir, srcdir, passwords,
        get_container_name, extra_containers, log_syslog=False):
    """
    Start all supporting containers (containers required for CKAN to
    operate) if they aren't already running, along with some extra
    containers specified by the user
    """
    if docker.is_boot2docker():
        docker.data_only_container(get_container_name('pgdata'),
            ['/var/lib/postgresql/data'])
        rw = {}
        volumes_from = get_container_name('pgdata')
    else:
        rw = {sitedir + '/postgres': '/var/lib/postgresql/data'}
        volumes_from = None

    running = set(containers_running(get_container_name))

    needed = set(extra_containers).union({'postgres', 'solr'})

    if not needed.issubset(running):
        stop_supporting_containers(get_container_name, extra_containers)

        # users are created when data dir is blank so we must pass
        # all the user passwords as environment vars
        # XXX: postgres entrypoint magic
        docker.run_container(
            name=get_container_name('postgres'),
            image='datacats/postgres',
            environment=passwords,
            rw=rw,
            volumes_from=volumes_from,
            log_syslog=log_syslog)

        docker.run_container(
            name=get_container_name('solr'),
            image='datacats/solr',
            rw={sitedir + '/solr': '/var/lib/solr'},
            ro={srcdir + '/schema.xml': '/etc/solr/conf/schema.xml'},
            log_syslog=log_syslog)

        for container in extra_containers:
            # We don't know a whole lot about the extra containers so we're just gonna have to
            # mount /project and /datadir r/o even if they're not needed for ease of
            # implementation.
            docker.run_container(
                name=get_container_name(container),
                image=EXTRA_IMAGE_MAPPING[container],
                ro={
                    sitedir: '/datadir',
                    srcdir: '/project'
                },
                log_syslog=log_syslog
            )
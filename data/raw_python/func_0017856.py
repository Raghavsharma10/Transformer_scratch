def _machine_check_connectivity():
    """
    This method calls to docker-machine on the command line and
    makes sure that it is up and ready.

    Potential improvements to be made:
        - Support multiple machine names (run a `docker-machine ls` and then
        see which machines are active. Use a priority list)
    """
    with open(devnull, 'w') as devnull_f:
        try:
            status = subprocess.check_output(
                ['docker-machine', 'status', 'dev'],
                stderr=devnull_f).strip()
            if status == 'Stopped':
                raise DatacatsError('Please start your docker-machine '
                                    'VM with "docker-machine start dev"')

            # XXX HACK: This exists because of
            #           http://github.com/datacats/datacats/issues/63,
            # as a temporary fix.
            if 'tls' in _docker_kwargs:
                # It will print out messages to the user otherwise.
                _docker_kwargs['tls'].assert_hostname = False
        except subprocess.CalledProcessError:
            raise DatacatsError('Please create a docker-machine with '
                                '"docker-machine start dev"')
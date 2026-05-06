def web_command(command, ro=None, rw=None, links=None,
                image='datacats/web', volumes_from=None, commit=False,
                clean_up=False, stream_output=None, entrypoint=None):
    """
    Run a single command in a web image optionally preloaded with the ckan
    source and virtual envrionment.

    :param command: command to execute
    :param ro: {localdir: binddir} dict for read-only volumes
    :param rw: {localdir: binddir} dict for read-write volumes
    :param links: links passed to start
    :param image: docker image name to use
    :param volumes_from:
    :param commit: True to create a new image based on result
    :param clean_up: True to remove container even on error
    :param stream_output: file to write stderr+stdout from command
    :param entrypoint: override entrypoint (script that runs command)

    :returns: image id if commit=True
    """
    binds = ro_rw_to_binds(ro, rw)
    c = _get_docker().create_container(
        image=image,
        command=command,
        volumes=binds_to_volumes(binds),
        detach=False,
        host_config=_get_docker().create_host_config(binds=binds, volumes_from=volumes_from, links=links),
        entrypoint=entrypoint)
    _get_docker().start(
        container=c['Id'],
        )
    if stream_output:
        for output in _get_docker().attach(
                c['Id'], stdout=True, stderr=True, stream=True):
            stream_output.write(output)
    if _get_docker().wait(c['Id']):
        # Before the (potential) cleanup, grab the logs!
        logs = _get_docker().logs(c['Id'])

        if clean_up:
            remove_container(c['Id'])
        raise WebCommandError(command, c['Id'][:12], logs)
    if commit:
        rval = _get_docker().commit(c['Id'])
    if not remove_container(c['Id']):
        # circle ci doesn't let us remove containers, quiet the warnings
        if not environ.get('CIRCLECI', False):
            warn('failed to remove container: {0}'.format(c['Id']))
    if commit:
        return rval['Id']
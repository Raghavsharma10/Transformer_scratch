def run_container(name, image, command=None, environment=None,
                  ro=None, rw=None, links=None, detach=True, volumes_from=None,
                  port_bindings=None, log_syslog=False):
    """
    Wrapper for docker create_container, start calls

    :param log_syslog: bool flag to redirect container's logs to host's syslog

    :returns: container info dict or None if container couldn't be created

    Raises PortAllocatedError if container couldn't start on the
    requested port.
    """
    binds = ro_rw_to_binds(ro, rw)
    log_config = LogConfig(type=LogConfig.types.JSON)
    if log_syslog:
        log_config = LogConfig(
            type=LogConfig.types.SYSLOG,
            config={'syslog-tag': name})
    host_config = _get_docker().create_host_config(binds=binds, log_config=log_config, links=links, volumes_from=volumes_from, port_bindings=port_bindings)

    c = _get_docker().create_container(
        name=name,
        image=image,
        command=command,
        environment=environment,
        volumes=binds_to_volumes(binds),
        detach=detach,
        stdin_open=False,
        tty=False,
        ports=list(port_bindings) if port_bindings else None,
        host_config=host_config)
    try:
        _get_docker().start(
            container=c['Id'],
        )
    except APIError as e:
        if 'address already in use' in e.explanation:
            try:
                _get_docker().remove_container(name, force=True)
            except APIError:
                pass
            raise PortAllocatedError()
        raise
    return c
def disable_openssh_rdns(distribution):
    """
    Set 'UseDNS no' in openssh config to disable rDNS lookups

    On each request for a new channel openssh defaults to an
    rDNS lookup on the client IP. This can be slow, if it fails
    for instance, adding 10s of overhead to every request
    for a new channel (not connection). This can add a lot of
    time to a process that opens lots of channels (e.g. running
    several commands via fabric.)

    This function will disable rDNS lookups in the openssh
    config and reload ssh to adjust the running instance.

    :param bytes distribution: the name of the distribution
        running on the node.
    """
    log_green('Disabling openssh reverse dns lookups')
    openssh_config_file = '/etc/ssh/sshd_config'
    dns_config = 'UseDNS no'
    if not file_contains(openssh_config_file, dns_config, use_sudo=True):
        file_append(openssh_config_file, dns_config, use_sudo=True)
        service_name = 'sshd'
        if 'ubuntu' in distribution:
            service_name = 'ssh'
        sudo('service {} reload'.format(service_name))
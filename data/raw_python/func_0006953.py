def print_gce_info(zone, project, instance_name, data):
    """ outputs information about our Rackspace instance """
    try:
        instance_info = _get_gce_compute().instances().get(
            project=project,
            zone=zone,
            instance=instance_name
        ).execute()
        log_yellow(pformat(instance_info))
        log_green("Instance state: %s" % instance_info['status'])
        log_green("Ip address: %s" % data['ip_address'])
    except HttpError as e:
        if e.resp.status != 404:
            raise e
        log_yellow("Instance state: DOWN")
    log_green("project: %s" % project)
    log_green("zone: %s" % zone)
    log_green("disk_name: %s" % instance_name)
    log_green("user: %s" % data['username'])
    log_green("ssh -i %s %s@%s" % (env.key_filename,
                                   data['username'],
                                   data['ip_address']))
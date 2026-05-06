def startup_gce_instance(instance_name, project, zone, username, machine_type,
                         image, public_key, disk_name=None):
    """
    For now, jclouds is broken for GCE and we will have static slaves
    in Jenkins.  Use this to boot them.
    """
    log_green("Started...")
    log_yellow("...Creating GCE Jenkins Slave Instance...")
    instance_config = get_gce_instance_config(
        instance_name, project, zone, machine_type, image,
        username, public_key, disk_name
    )
    operation = _get_gce_compute().instances().insert(
        project=project,
        zone=zone,
        body=instance_config
    ).execute()
    result = gce_wait_until_done(operation)
    if not result:
        raise RuntimeError("Creation of VM timed out or returned no result")
    log_green("Instance has booted")
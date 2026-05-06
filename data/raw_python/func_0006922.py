def create_gce_image(zone,
                     project,
                     instance_name,
                     name,
                     description):
    """
    Shuts down the instance and creates and image from the disk.

    Assumes that the disk name is the same as the instance_name (this is the
    default behavior for boot disks on GCE).
    """

    disk_name = instance_name
    try:
        down_gce(instance_name=instance_name, project=project, zone=zone)
    except HttpError as e:
        if e.resp.status == 404:
            log_yellow("the instance {} is already down".format(instance_name))
        else:
            raise e

    body = {
        "rawDisk": {},
        "name": name,
        "sourceDisk": "projects/{}/zones/{}/disks/{}".format(
            project, zone, disk_name
        ),
        "description": description
    }
    compute = _get_gce_compute()
    gce_wait_until_done(
        compute.images().insert(project=project, body=body).execute()
    )
    return name
def get_region():
    """Gets the AWS Region ID for this system

    :return: (str) AWS Region ID where this system lives
    """
    log = logging.getLogger(mod_logger + '.get_region')

    # First get the availability zone
    availability_zone = get_availability_zone()

    if availability_zone is None:
        msg = 'Unable to determine the Availability Zone for this system, cannot determine the AWS Region'
        log.error(msg)
        return

    # Strip of the last character to get the region
    region = availability_zone[:-1]
    return region
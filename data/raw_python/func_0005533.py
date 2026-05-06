def get_availability_zone():
    """Gets the AWS Availability Zone ID for this system

    :return: (str) Availability Zone ID where this system lives
    """
    log = logging.getLogger(mod_logger + '.get_availability_zone')

    # Exit if not running on AWS
    if not is_aws():
        log.info('This machine is not running in AWS, exiting...')
        return

    availability_zone_url = metadata_url + 'placement/availability-zone'
    try:
        response = urllib.urlopen(availability_zone_url)
    except(IOError, OSError) as ex:
        msg = 'Unable to query URL to get Availability Zone: {u}\n{e}'.format(u=availability_zone_url, e=ex)
        log.error(msg)
        return

    # Check the code
    if response.getcode() != 200:
        msg = 'There was a problem querying url: {u}, returned code: {c}, unable to get the Availability Zone'.format(
            u=availability_zone_url, c=response.getcode())
        log.error(msg)
        return
    availability_zone = response.read()
    return availability_zone
def get_vpc_id_from_mac_address():
    """Gets the VPC ID for this EC2 instance

    :return: String instance ID or None
    """
    log = logging.getLogger(mod_logger + '.get_vpc_id')

    # Exit if not running on AWS
    if not is_aws():
        log.info('This machine is not running in AWS, exiting...')
        return

    # Get the primary interface MAC address to query the meta data service
    log.debug('Attempting to determine the primary interface MAC address...')
    try:
        mac_address = get_primary_mac_address()
    except AWSMetaDataError:
        _, ex, trace = sys.exc_info()
        msg = '{n}: Unable to determine the mac address, cannot determine VPC ID:\n{e}'.format(
            n=ex.__class__.__name__, e=str(ex))
        log.error(msg)
        return

    vpc_id_url = metadata_url + 'network/interfaces/macs/' + mac_address + '/vpc-id'
    try:
        response = urllib.urlopen(vpc_id_url)
    except(IOError, OSError) as ex:
        msg = 'Unable to query URL to get VPC ID: {u}\n{e}'.format(u=vpc_id_url, e=ex)
        log.error(msg)
        return

    # Check the code
    if response.getcode() != 200:
        msg = 'There was a problem querying url: {u}, returned code: {c}, unable to get the vpc-id'.format(
                u=vpc_id_url, c=response.getcode())
        log.error(msg)
        return
    vpc_id = response.read()
    return vpc_id
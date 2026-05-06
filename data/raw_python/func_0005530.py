def is_aws():
    """Determines if this system is on AWS

    :return: bool True if this system is running on AWS
    """
    log = logging.getLogger(mod_logger + '.is_aws')
    log.info('Querying AWS meta data URL: {u}'.format(u=metadata_url))

    # Re-try logic for checking the AWS meta data URL
    retry_time_sec = 10
    max_num_tries = 10
    attempt_num = 1

    while True:
        if attempt_num > max_num_tries:
            log.info('Unable to query the AWS meta data URL, this system is NOT running on AWS\n{e}')
            return False

        # Query the AWS meta data URL
        try:
            response = urllib.urlopen(metadata_url)
        except(IOError, OSError) as ex:
            log.warn('Failed to query the AWS meta data URL\n{e}'.format(e=str(ex)))
            attempt_num += 1
            time.sleep(retry_time_sec)
            continue

        # Check the code
        if response.getcode() == 200:
            log.info('AWS metadata service returned code 200, this system is running on AWS')
            return True
        else:
            log.warn('AWS metadata service returned code: {c}'.format(c=response.getcode()))
            attempt_num += 1
            time.sleep(retry_time_sec)
            continue
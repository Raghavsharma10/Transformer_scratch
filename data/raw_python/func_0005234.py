def find_bucket_keys(bucket_name, regex, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
    """Finds a list of S3 keys matching the passed regex

    Given a regular expression, this method searches the S3 bucket
    for matching keys, and returns an array of strings for matched
    keys, an empty array if non are found.

    :param regex: (str) Regular expression to use is the key search
    :param bucket_name: (str) String S3 bucket name
    :param region_name: (str) AWS region for the S3 bucket (optional)
    :param aws_access_key_id: (str) AWS Access Key ID (optional)
    :param aws_secret_access_key: (str) AWS Secret Access Key (optional)
    :return: Array of strings containing matched S3 keys
    """
    log = logging.getLogger(mod_logger + '.find_bucket_keys')
    matched_keys = []
    if not isinstance(regex, basestring):
        log.error('regex argument is not a string, found: {t}'.format(t=regex.__class__.__name__))
        return None
    if not isinstance(bucket_name, basestring):
        log.error('bucket_name argument is not a string, found: {t}'.format(t=bucket_name.__class__.__name__))
        return None

    # Set up S3 resources
    s3resource = boto3.resource('s3', region_name=region_name, aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key)
    bucket = s3resource.Bucket(bucket_name)

    log.info('Looking up S3 keys based on regex: {r}'.format(r=regex))
    for item in bucket.objects.all():
        log.debug('Checking if regex matches key: {k}'.format(k=item.key))
        match = re.search(regex, item.key)
        if match:
            matched_keys.append(item.key)
    log.info('Found matching keys: {k}'.format(k=matched_keys))
    return matched_keys
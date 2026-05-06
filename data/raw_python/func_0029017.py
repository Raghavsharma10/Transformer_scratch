def ls(awsclient, bucket, prefix=None):
    """List bucket contents

    :param awsclient:
    :param bucket:
    :param prefix:
    :return:
    """
    # this works until 1000 keys!
    params = {'Bucket': bucket}
    if prefix:
        params['Prefix'] = prefix
    client_s3 = awsclient.get_client('s3')
    objects = client_s3.list_objects_v2(**params)
    if objects['KeyCount'] > 0:
        keys = [k['Key'] for k in objects['Contents']]
        return keys
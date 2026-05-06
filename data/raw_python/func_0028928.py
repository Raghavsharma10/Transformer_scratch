def get_base_ami(awsclient, owners):
    """
    DEPRECATED!!!
    return the latest version of our base AMI
    we can't use tags for this, so we have only the name as resource
    note: this functionality is deprecated since this only works for "old"
    baseami. 
    """
    client_ec2 = awsclient.get_client('ec2')
    image_filter = [
        {
            'Name': 'state',
            'Values': [
                'available',
            ]
        },
    ]

    latest_ts = maya.MayaDT(0).datetime(naive=True)
    latest_version = StrictVersion('0.0.0')
    latest_id = None
    for i in client_ec2.describe_images(
            Owners=owners,
            Filters=image_filter
            )['Images']:
        m = re.search(r'(Ops_Base-Image)_(\d+.\d+.\d+)_(\d+)$', i['Name'])
        if m:
            version = StrictVersion(m.group(2))
            #timestamp = m.group(3)
            creation_date = parse_ts(i['CreationDate'])

            if creation_date > latest_ts and version >=latest_version:
                latest_id = i['ImageId']
                latest_ts = creation_date
                latest_version = version

    return latest_id
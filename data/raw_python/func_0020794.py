def deploy(
    config,
    name,
    bucket,
    timeout,
    memory,
    description,
    subnet_ids,
    security_group_ids
):
    """ Deploy/Update a function from a project directory """
    # options should override config if it is there
    myname = name or config.name
    mybucket = bucket or config.bucket
    mytimeout = timeout or config.timeout
    mymemory = memory or config.memory
    mydescription = description or config.description
    mysubnet_ids = subnet_ids or config.subnet_ids
    mysecurity_group_ids = security_group_ids or config.security_group_ids

    vpc_config = {}
    if mysubnet_ids and mysecurity_group_ids:
        vpc_config = {
            'SubnetIds': mysubnet_ids.split(','),
            'SecurityGroupIds': mysecurity_group_ids.split(',')
        }

    click.echo('Deploying {} to {}'.format(myname, mybucket))
    lambder.deploy_function(
        myname,
        mybucket,
        mytimeout,
        mymemory,
        mydescription,
        vpc_config
    )
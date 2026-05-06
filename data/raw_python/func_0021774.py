def list_ec2(region, filter_by_kwargs):
    """List running ec2 instances."""
    conn = boto.ec2.connect_to_region(region)
    instances = conn.get_only_instances()
    return lookup(instances, filter_by=filter_by_kwargs)
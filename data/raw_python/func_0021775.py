def list_ebs(region, filter_by_kwargs):
    """List running ebs volumes."""
    conn = boto.ec2.connect_to_region(region)
    instances = conn.get_all_volumes()
    return lookup(instances, filter_by=filter_by_kwargs)
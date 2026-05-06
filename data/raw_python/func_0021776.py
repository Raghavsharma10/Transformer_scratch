def list_elb(region, filter_by_kwargs):
    """List all load balancers."""
    conn = boto.ec2.elb.connect_to_region(region)
    instances = conn.get_all_load_balancers()
    return lookup(instances, filter_by=filter_by_kwargs)
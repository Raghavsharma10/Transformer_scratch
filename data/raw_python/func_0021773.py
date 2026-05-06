def list_cloudfront(region, filter_by_kwargs):
    """List running ec2 instances."""
    conn = boto.connect_cloudfront()
    instances = conn.get_all_distributions()
    return lookup(instances, filter_by=filter_by_kwargs)
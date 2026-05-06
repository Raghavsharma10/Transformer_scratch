def list_autoscaling_group(region, filter_by_kwargs):
    """List all Auto Scaling Groups."""
    conn = boto.ec2.autoscale.connect_to_region(region)
    groups = conn.get_all_groups()
    return lookup(groups, filter_by=filter_by_kwargs)
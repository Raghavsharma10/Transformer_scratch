def list_rds(region, filter_by_kwargs):
    """List all RDS thingys."""
    conn = boto.rds.connect_to_region(region)
    instances = conn.get_all_dbinstances()
    return lookup(instances, filter_by=filter_by_kwargs)
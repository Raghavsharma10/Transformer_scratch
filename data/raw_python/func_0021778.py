def list_elasticache(region, filter_by_kwargs):
    """List all ElastiCache Clusters."""
    conn = boto.elasticache.connect_to_region(region)
    req = conn.describe_cache_clusters()
    data = req["DescribeCacheClustersResponse"]["DescribeCacheClustersResult"]["CacheClusters"]
    if filter_by_kwargs:
        clusters = [x['CacheClusterId'] for x in data if x[filter_by_kwargs.keys()[0]] == filter_by_kwargs.values()[0]]
    else:
        clusters = [x['CacheClusterId'] for x in data]
    return clusters
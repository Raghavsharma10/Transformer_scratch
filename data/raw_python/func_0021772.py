def list_billing(region, filter_by_kwargs):
    """List available billing metrics"""
    conn = boto.ec2.cloudwatch.connect_to_region(region)
    metrics = conn.list_metrics(metric_name='EstimatedCharges')
    # Filtering is based on metric Dimensions.  Only really valuable one is
    # ServiceName.
    if filter_by_kwargs:
        filter_key = filter_by_kwargs.keys()[0]
        filter_value = filter_by_kwargs.values()[0]
        if filter_value:
            filtered_metrics = [x for x in metrics if x.dimensions.get(filter_key) and x.dimensions.get(filter_key)[0] == filter_value]
        else:
            # ServiceName=''
            filtered_metrics = [x for x in metrics if not x.dimensions.get(filter_key)]
    else:
        filtered_metrics = metrics
    return filtered_metrics
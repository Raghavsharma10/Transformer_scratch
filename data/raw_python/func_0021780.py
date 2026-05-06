def list_sqs(region, filter_by_kwargs):
    """List all SQS Queues."""
    conn = boto.sqs.connect_to_region(region)
    queues = conn.get_all_queues()
    return lookup(queues, filter_by=filter_by_kwargs)
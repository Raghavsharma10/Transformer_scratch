def list_dynamodb(region, filter_by_kwargs):
    """List all DynamoDB tables."""
    conn = boto.dynamodb.connect_to_region(region)
    tables = conn.list_tables()
    return lookup(tables, filter_by=filter_by_kwargs)
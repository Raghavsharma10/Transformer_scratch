def ls(args):
    """
    List S3 buckets. See also "aws s3 ls". Use "aws s3 ls NAME" to list bucket contents.
    """
    table = []
    for bucket in filter_collection(resources.s3.buckets, args):
        bucket.LocationConstraint = clients.s3.get_bucket_location(Bucket=bucket.name)["LocationConstraint"]
        cloudwatch = resources.cloudwatch
        bucket_region = bucket.LocationConstraint or "us-east-1"
        if bucket_region != cloudwatch.meta.client.meta.region_name:
            cloudwatch = boto3.Session(region_name=bucket_region).resource("cloudwatch")
        data = get_cloudwatch_metric_stats("AWS/S3", "NumberOfObjects",
                                           start_time=datetime.utcnow() - timedelta(days=2),
                                           end_time=datetime.utcnow(), period=3600, BucketName=bucket.name,
                                           StorageType="AllStorageTypes", resource=cloudwatch)
        bucket.NumberOfObjects = int(data["Datapoints"][-1]["Average"]) if data["Datapoints"] else None
        data = get_cloudwatch_metric_stats("AWS/S3", "BucketSizeBytes",
                                           start_time=datetime.utcnow() - timedelta(days=2),
                                           end_time=datetime.utcnow(), period=3600, BucketName=bucket.name,
                                           StorageType="StandardStorage", resource=cloudwatch)
        bucket.BucketSizeBytes = format_number(data["Datapoints"][-1]["Average"]) if data["Datapoints"] else None
        table.append(bucket)
    page_output(tabulate(table, args))
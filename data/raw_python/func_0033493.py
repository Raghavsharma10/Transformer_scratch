def get_bucket(s3_bucket=None, validate=False):
    """Gets a bucket from specified settings"""
    global S3Connection

    if S3Connection != None:
        settings = oz.settings
        s3_bucket = s3_bucket or settings["s3_bucket"]
        opts = {}
        if settings["s3_host"]:
            opts["host"] = settings["s3_host"]
        if settings["aws_access_key"] and settings["aws_secret_key"]:
            opts["aws_access_key_id"] = settings["aws_access_key"]
            opts["aws_secret_access_key"] = settings["aws_secret_key"]
        return S3Connection(**opts).get_bucket(s3_bucket, validate=validate)
    else:
        raise Exception("S3 not supported in this environment as boto is not installed")
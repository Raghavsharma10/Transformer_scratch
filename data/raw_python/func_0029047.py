def _read_s3_config(self):
        """Read in the value of the configuration file in Amazon S3.

        :rtype: str
        :raises: ValueError

        """
        try:
            import boto3
            import botocore.exceptions
        except ImportError:
            boto3, botocore = None, None

        if not boto3:
            raise ValueError(
                's3 URL specified for configuration but boto3 not installed')
        parsed = parse.urlparse(self._file_path)
        try:
            response = boto3.client(
                's3', endpoint_url=os.environ.get('S3_ENDPOINT')).get_object(
                    Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
        except botocore.exceptions.ClientError as e:
            raise ValueError(
                'Failed to download configuration from S3: {}'.format(e))
        return response['Body'].read().decode('utf-8')
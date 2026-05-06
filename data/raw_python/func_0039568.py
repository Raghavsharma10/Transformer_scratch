def list_dir(self):
        """
        Non-recursive file listing.

        :returns: A generator over files in this "directory" for efficiency.
        """

        bucket = self.s3_object.Bucket()
        prefix = self.s3_object.key
        if not prefix.endswith('/'): prefix += '/'

        for obj in bucket.objects.filter(Delimiter='/', Prefix=prefix):
            yield 's3://{}/{}'.format(obj.bucket_name, obj.key)
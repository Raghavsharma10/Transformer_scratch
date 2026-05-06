def list_dir(self):
        """
        Non-recursive file listing.

        :returns: A generator over files in this "directory" for efficiency.
        """

        bucket = self.blob.bucket
        prefix = self.blob.name
        if not prefix.endswith('/'): prefix += '/'

        for blob in bucket.list_blobs(prefix=prefix, delimiter='/'):
            yield 'gs://{}/{}'.format(blob.bucket.name, blob.name)
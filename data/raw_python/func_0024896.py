def list_objects(self, bucket_name=None, **kwargs):
        """
        This method is primarily for illustration and just calls the 
        boto3 client implementation of list_objects but is a common task
        for first time Predix BlobStore users.
        """
        if not bucket_name: bucket_name = self.bucket_name
        return self.client.list_objects(Bucket=bucket_name, **kwargs)
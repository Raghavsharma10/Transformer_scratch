def upload_file(self, src_filepath, dest_filename=None, bucket_name=None,
            **kwargs):
        """
        This method is primarily for illustration and just calls the 
        boto3 client implementation of upload_file but is a common task
        for first time Predix BlobStore users.
        """
        if not bucket_name: bucket_name = self.bucket_name
        if not dest_filename: dest_filename = src_filepath
        return self.client.upload_file(src_filepath, bucket_name,
                dest_filename, **kwargs)
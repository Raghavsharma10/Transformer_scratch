def upload_file(self, filepath, key):
        """Uploads a file using the passed S3 key

        This method uploads a file specified by the filepath to S3
        using the provided S3 key.

        :param filepath: (str) Full path to the file to be uploaded
        :param key: (str) S3 key to be set for the upload
        :return: True if upload is successful, False otherwise.
        """
        log = logging.getLogger(self.cls_logger + '.upload_file')
        log.info('Attempting to upload file %s to S3 bucket %s as key %s...',
                 filepath, self.bucket_name, key)

        if not isinstance(filepath, basestring):
            log.error('filepath argument is not a string')
            return False

        if not isinstance(key, basestring):
            log.error('key argument is not a string')
            return False

        if not os.path.isfile(filepath):
            log.error('File not found on file system: %s', filepath)
            return False

        try:
            self.s3client.upload_file(
                Filename=filepath, Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            log.error('Unable to upload file %s to bucket %s as key %s:\n%s',
                      filepath, self.bucket_name, key, e)
            return False
        else:
            log.info('Successfully uploaded file to S3 bucket %s as key %s',
                     self.bucket_name, key)
            return True
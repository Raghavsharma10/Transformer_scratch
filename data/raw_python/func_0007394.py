def _upload_s3(self, zip_file):
        '''
        Uploads the lambda package to s3
        '''
        s3_client = self._aws_session.client('s3')
        transfer = boto3.s3.transfer.S3Transfer(s3_client)
        transfer.upload_file(zip_file, self._config.s3_bucket,
                             self._config.s3_package_name())
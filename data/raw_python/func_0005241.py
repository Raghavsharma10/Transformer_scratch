def delete_key(self, key_to_delete):
        """Deletes the specified key

        :param key_to_delete:
        :return:
        """
        log = logging.getLogger(self.cls_logger + '.delete_key')

        log.info('Attempting to delete key: {k}'.format(k=key_to_delete))
        try:
            self.s3client.delete_object(Bucket=self.bucket_name, Key=key_to_delete)
        except ClientError:
            _, ex, trace = sys.exc_info()
            log.error('ClientError: Unable to delete key: {k}\n{e}'.format(k=key_to_delete, e=str(ex)))
            return False
        else:
            log.info('Successfully deleted key: {k}'.format(k=key_to_delete))
            return True
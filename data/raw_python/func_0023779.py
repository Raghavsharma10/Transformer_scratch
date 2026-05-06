def del_stream(self, bucket, label):
        """ Will fail if the bucket or label don't exist """
        bucket = self._require_bucket(bucket)
        key = self._require_key(bucket, label)
        key.delete()
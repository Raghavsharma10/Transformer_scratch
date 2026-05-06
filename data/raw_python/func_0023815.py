def put_stream(self, bucket, label, stream_object, params={}):
        ''' Create a new file to swift object storage. '''
        self.claim_bucket(bucket) 
        self.connection.put_object(bucket, label, stream_object,
                                   headers=self._convert_to_meta(params))
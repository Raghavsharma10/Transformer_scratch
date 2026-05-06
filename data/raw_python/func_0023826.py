def get_metadata(self, bucket, label):
        '''Get the metadata for this bucket:label identifier.
        '''
        if self.mode !="w":
            try:
                jsn = self._get_bucket_md(bucket)
            except OFSFileNotFound:
                # No MD found...
                return {}
            except OFSException as e:
                raise OFSException(e)
            if label in jsn:
                return jsn[label]
            else:
                return {}
        else:
            raise OFSException("Cannot read md from archive in 'w' mode")
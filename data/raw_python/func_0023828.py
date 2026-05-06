def del_metadata_keys(self, bucket, label, keys):
        '''Delete the metadata corresponding to the specified keys.
        '''
        if self.mode !="r":
            try:
                payload = self._get_bucket_md(bucket)
            except OFSFileNotFound:
                # No MD found...
                raise OFSFileNotFound("Couldn't find a md file for %s bucket" % bucket)
            except OFSException as e:
                raise OFSException(e)
            if payload.has_key(label):
                for key in [x for x in keys if payload[label].has_key(x)]:
                    del payload[label][key]
            self.put_stream(bucket, MD_FILE, json.dumps(payload), params={}, replace=True, add_md=False)
        else:
            raise OFSException("Cannot update MD in archive in 'r' mode")
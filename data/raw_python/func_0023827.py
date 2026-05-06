def update_metadata(self, bucket, label, params):
        '''Update the metadata with the provided dictionary of params.

        :param parmams: dictionary of key values (json serializable).
        '''
        if self.mode !="r":
            try:
                payload = self._get_bucket_md(bucket)
            except OFSFileNotFound:
                # No MD found... create it
                payload = {}
                for l in self.list_labels(bucket):
                    payload[l] = {}
                    payload[l]['_label'] = l
                if not self.quiet:
                    print("Had to create md file for %s" % bucket)
            except OFSException as e:
                raise OFSException(e)
            if not label in payload:
                payload[label] = {}
            payload[label].update(params)
            self.put_stream(bucket, MD_FILE, json.dumps(payload).encode('utf-8'), params={}, replace=True, add_md=False)
            return payload[label]
        else:
            raise OFSException("Cannot update MD in archive in 'r' mode")
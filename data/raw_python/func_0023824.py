def put_stream(self, bucket, label, stream_object, params=None, replace=True, add_md=True):
        '''Put a bitstream (stream_object) for the specified bucket:label identifier.

        :param bucket: as standard
        :param label: as standard
        :param stream_object: file-like object to read from or bytestring.
        :param params: update metadata with these params (see `update_metadata`)
        '''
        if self.mode == "r":
            raise OFSException("Cannot write into archive in 'r' mode")
        else:
            params = params or {}
            fn = self._zf(bucket, label)
            params['_creation_date'] = datetime.now().isoformat().split(".")[0]  ## '2010-07-08T19:56:47'
            params['_label'] = label
            if self.exists(bucket, label) and replace==True:
                # Add then Replace? Let's see if that works...
                #z = ZipFile(self.zipfile, self.mode, self.compression, self.allowZip64)
                zinfo = self.z.getinfo(fn)
                size, chksum = self._write(self.z, bucket, label, stream_object)
                self._del_stream(zinfo)
                #z.close()
                params['_content_length'] = size
                if chksum:
                    params['_checksum'] = chksum
            else:
                #z = ZipFile(self.zipfile, self.mode, self.compression, self.allowZip64)
                size, chksum = self._write(self.z, bucket, label, stream_object)
                #z.close()
                params['_content_length'] = size
                if chksum:
                    params['_checksum'] = chksum
            if add_md:
                params = self.update_metadata(bucket, label, params)
            return params
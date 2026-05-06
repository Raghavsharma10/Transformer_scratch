def get_stream(self, bucket, label, as_stream=True):
        '''Get a bitstream for the given bucket:label combination.

        :param bucket: the bucket to use.
        :return: bitstream as a file-like object
        '''
        if self.mode == "w":
            raise OFSException("Cannot read from archive in 'w' mode")
        elif self.exists(bucket, label):
            fn = self._zf(bucket, label)
            if as_stream:
                return self.z.open(fn)
            else:
                return self.z.read(fn)
        else:
            raise OFSFileNotFound
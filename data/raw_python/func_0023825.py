def del_stream(self, bucket, label):
        '''Delete a bitstream. This needs more testing - file deletion in a zipfile
        is problematic. Alternate method is to create second zipfile without the files
        in question, which is not a nice method for large zip archives.
        '''
        if self.exists(bucket, label):
            name = self._zf(bucket, label)
            #z = ZipFile(self.zipfile, self.mode, self.compression, self.allowZip64)
            self._del_stream(name)
def get_url(self, bucket, label):
        '''Get a URL that should point at the bucket:labelled resource. Aimed to aid web apps by allowing them to redirect to an open resource, rather than proxy the bitstream.

        :param bucket: the bucket to use.
        :param label: the label of the resource to get
        :return: a string URI - eg 'zip:file:///home/.../foo.zip!/bucket/label'
        '''
        if self.exists(bucket, label):
            root = "zip:file//%s" % os.path.abspath(self.zipfile)
            fn = self._zf(bucket, label)
            return "!/".join(root, fn)
        else:
            raise OFSFileNotFound
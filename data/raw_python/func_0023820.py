def list_labels(self, bucket):
        '''List labels for the given bucket. Due to zipfiles inherent arbitrary ordering,
        this is an expensive operation, as it walks the entire archive searching for individual
        'buckets'

        :param bucket: bucket to list labels for.
        :return: iterator for the labels in the specified bucket.
        '''
        for name in self.z.namelist():
            container, label = self._nf(name.encode("utf-8"))
            if container == bucket and label != MD_FILE:
                yield label
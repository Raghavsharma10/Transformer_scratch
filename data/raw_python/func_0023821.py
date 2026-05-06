def list_buckets(self):
        '''List all buckets managed by this OFS instance. Like list_labels, this also
        walks the entire archive, yielding the bucketnames. A local set is retained so that
        duplicates aren't returned so this will temporarily pull the entire list into memory
        even though this is a generator and will slow as more buckets are added to the set.

        :return: iterator for the buckets.
        '''
        buckets = set()
        for name in self.z.namelist():
            bucket, _ = self._nf(name)
            if bucket not in buckets:
                buckets.add(bucket)
                yield bucket
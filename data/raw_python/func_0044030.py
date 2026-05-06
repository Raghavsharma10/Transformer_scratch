def list_from_metadata(cls, url, metadata):
        '''return a list of DatalakeRecords for the url and metadata'''
        key = cls._get_key(url)
        metadata = Metadata(**metadata)
        ct = cls._get_create_time(key)
        time_buckets = cls.get_time_buckets_from_metadata(metadata)
        return [cls(url, metadata, t, ct, key.size) for t in time_buckets]
def list_from_url(cls, url):
        '''return a list of DatalakeRecords for the specified url'''
        key = cls._get_key(url)
        metadata = cls._get_metadata_from_key(key)
        ct = cls._get_create_time(key)
        time_buckets = cls.get_time_buckets_from_metadata(metadata)
        return [cls(url, metadata, t, ct, key.size) for t in time_buckets]
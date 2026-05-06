def make_label(self, path):
        """
        this borrows too much from the internals of ofs
        maybe expose different parts of the api?
        """
        from datetime import datetime
        from StringIO import StringIO
        path = path.lstrip("/")
        bucket, label = path.split("/", 1)

        bucket = self.ofs._require_bucket(bucket)
        key = self.ofs._get_key(bucket, label)
        if key is None:
            key = bucket.new_key(label)
            self.ofs._update_key_metadata(key, { '_creation_time': str(datetime.utcnow()) })
            key.set_contents_from_file(StringIO(''))
        key.close()
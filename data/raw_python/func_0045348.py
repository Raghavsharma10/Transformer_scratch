def _write_version(self, data, model):
        """
            Writes a copy of the objects current state to write-once mirror bucket.

        Args:
            data (dict): Model instance's all data for versioning.
            model (instance): Model instance.

        Returns:
            Key of version record.
            key (str): Version_bucket key.
        """
        vdata = {'data': data,
                 'key': model.key,
                 'model': model.Meta.bucket_name,
                 'timestamp': time.time()}
        obj = version_bucket.new(data=vdata)
        obj.add_index('key_bin', model.key)
        obj.add_index('model_bin', vdata['model'])
        obj.add_index('timestamp_int', int(vdata['timestamp']))
        obj.store()
        return obj.key
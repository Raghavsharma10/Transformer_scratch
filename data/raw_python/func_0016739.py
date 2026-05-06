def _unpack(self, record, key, expected):
        """Replaces the attr dict at the given key with an instance of a Model"""
        attrs = record.get(key)
        if attrs is None:
            return
        obj = unpack_from_dynamodb(
            attrs=attrs,
            expected=expected,
            model=self.model,
            engine=self.engine
        )
        object_loaded.send(self.engine, engine=self.engine, obj=obj)
        record[key] = obj
def prepare_buckets(self):
        """
        loads buckets to bucket cache.
        """
        for mdl in self.registry.get_base_models():
            bucket = mdl(super_context).objects.adapter.bucket
            self.buckets[bucket.name] = bucket
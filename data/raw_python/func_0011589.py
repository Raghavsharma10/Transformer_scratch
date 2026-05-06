def update_instance(
            self, model_name, pk, instance=None, version=None,
            update_only=False):
        """Create or update a cached instance.

        Keyword arguments are:
        model_name - The name of the model
        pk - The primary key of the instance
        instance - The Django model instance, or None to load it
        versions - Version to update, or None for all
        update_only - If False (default), then missing cache entries will be
            populated and will cause follow-on invalidation.  If True, then
            only entries already in the cache will be updated and cause
            follow-on invalidation.

        Return is a list of tuples (model name, pk, immediate) that also needs
        to be updated.
        """
        versions = [version] if version else self.versions
        invalid = []
        for version in versions:
            serializer = self.model_function(model_name, version, 'serializer')
            loader = self.model_function(model_name, version, 'loader')
            invalidator = self.model_function(
                model_name, version, 'invalidator')
            if serializer is None and loader is None and invalidator is None:
                continue

            if self.cache is None:
                continue

            # Try to load the instance
            if not instance:
                instance = loader(pk)

            if serializer:
                # Get current value, if in cache
                key = self.key_for(version, model_name, pk)
                current_raw = self.cache.get(key)
                current = json.loads(current_raw) if current_raw else None

                # Get new value
                if update_only and current_raw is None:
                    new = None
                else:
                    new = serializer(instance)
                deleted = not instance

                # If cache is invalid, update cache
                invalidate = (current != new) or deleted
                if invalidate:
                    if deleted:
                        self.cache.delete(key)
                    else:
                        self.cache.set(key, json.dumps(new))
            else:
                invalidate = True

            # Invalidate upstream caches
            if instance and invalidate:
                for upstream in invalidator(instance):
                    if isinstance(upstream, str):
                        self.cache.delete(upstream)
                    else:
                        m, i, immediate = upstream
                        if immediate:
                            invalidate_key = self.key_for(version, m, i)
                            self.cache.delete(invalidate_key)
                        invalid.append((m, i, version))
        return invalid
def apply_async(self, args=None, kwargs=None, **options):
        """
        Checks whether task must be skipped and decreases the counter in that case.
        """
        key = self._get_cache_key(args, kwargs)
        counter, penalty = cache.get(key, (0, 0))
        if not counter:
            return super(PenalizedBackgroundTask, self).apply_async(args=args, kwargs=kwargs, **options)

        cache.set(key, (counter - 1, penalty), self.CACHE_LIFETIME)
        logger.info('The task %s will not be executed due to the penalty.' % self.name)
        return self.AsyncResult(options.get('task_id') or str(uuid4()))
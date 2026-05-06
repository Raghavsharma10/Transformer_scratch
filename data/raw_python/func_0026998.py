def on_success(self, retval, task_id, args, kwargs):
        """
        Clears cache for the task.
        """
        key = self._get_cache_key(args, kwargs)
        if cache.get(key) is not None:
            cache.delete(key)
            logger.debug('Penalty for the task %s has been removed.' % self.name)

        return super(PenalizedBackgroundTask, self).on_success(retval, task_id, args, kwargs)
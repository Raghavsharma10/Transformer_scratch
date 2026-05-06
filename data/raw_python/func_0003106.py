def get_status(self, instance):
        """Retrives a status of a field from cache. Fields in state 'error' and
        'complete' will not retain the status after the call.

        """
        status_key, status = self._get_status(instance)
        if status['state'] in ['complete', 'error']:
            cache.delete(status_key)
        return status
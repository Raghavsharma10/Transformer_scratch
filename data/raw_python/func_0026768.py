def delete_expired_locks(self):
        """
        Deletes all expired mutex locks if a ttl is provided.
        """
        ttl_seconds = self.get_mutex_ttl_seconds()
        if ttl_seconds is not None:
            DBMutex.objects.filter(creation_time__lte=timezone.now() - timedelta(seconds=ttl_seconds)).delete()
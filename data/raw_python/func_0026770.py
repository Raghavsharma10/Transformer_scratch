def stop(self):
        """
        Releases the db mutex lock. Throws an error if the lock was released before the function finished.
        """
        if not DBMutex.objects.filter(id=self.lock.id).exists():
            raise DBMutexTimeoutError('Lock {0} expired before function completed'.format(self.lock_id))
        else:
            self.lock.delete()
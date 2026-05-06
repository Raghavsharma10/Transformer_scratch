def start(self):
        """
        Acquires the db mutex lock. Takes the necessary steps to delete any stale locks.
        Throws a DBMutexError if it can't acquire the lock.
        """
        # Delete any expired locks first
        self.delete_expired_locks()
        try:
            with transaction.atomic():
                self.lock = DBMutex.objects.create(lock_id=self.lock_id)
        except IntegrityError:
            raise DBMutexError('Could not acquire lock: {0}'.format(self.lock_id))
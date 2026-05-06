def lock(self, key, owner, timeout=None, expire=None):
        """Lock given `key` to `owner`.

        :Parameters:
          - `key` - lock name
          - `owner` - name of application/component/whatever which asks for lock
          - `timeout` (optional) - how long to wait if `key` is locked
          - `expire` (optional) - when given, lock will be released after that number of seconds.

        Raises `MongoLockTimeout` if can't achieve a lock before timeout.
        """
        expire = datetime.utcnow() + timedelta(seconds=expire) if expire else None
        try:
            self.collection.insert({
                '_id': key,
                'locked': True,
                'owner': owner,
                'created': datetime.utcnow(),
                'expire': expire
            })
            return True
        except DuplicateKeyError:
            start_time = datetime.utcnow()
            while True:
                if self._try_get_lock(key, owner, expire):
                    return True

                if not timeout or datetime.utcnow() >= start_time + timedelta(seconds=timeout):
                    return False
                time.sleep(self.acquire_retry_step)
def release(self, key, owner):
        """Release lock with given name.
          `key` - lock name
          `owner` - name of application/component/whatever which held a lock
        Raises `MongoLockException` if no such a lock.
        """
        status = self.collection.find_and_modify(
            {'_id': key, 'owner': owner},
            {'locked': False, 'owner': None, 'created': None, 'expire': None}
        )
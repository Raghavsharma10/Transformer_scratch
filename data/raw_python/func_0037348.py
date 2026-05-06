def touch(self, key, owner, expire=None):
        """Renew lock to avoid expiration. """
        lock = self.collection.find_one({'_id': key, 'owner': owner})
        if not lock:
            raise MongoLockException(u'Can\'t find lock for {key}: {owner}'.format(key=key, owner=owner))
        if not lock['expire']:
            return
        if not expire:
            raise MongoLockException(u'Can\'t touch lock without expire for {0}: {1}'.format(key, owner))
        expire = datetime.utcnow() + timedelta(seconds=expire)
        self.collection.update(
            {'_id': key, 'owner': owner},
            {'$set': {'expire': expire}}
        )
def _random_key(self):
        """ Return random session key """
        hashstr = '%s%s' % (random.random(), self.time_module.time())
        return hashlib.md5(hashstr).hexdigest()
def incr(self, key, delta=1):
        """Increments the specified key value by the specified value.
       
        :param str|unicode key:
    
        :param int delta:

        :rtype: bool
        """
        return uwsgi.cache_inc(key, delta, self.timeout, self.name)
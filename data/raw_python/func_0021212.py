def _get_user_hash(self):
        """Calculate a digest based on request's User-Agent and IP address."""
        if request:
            user_hash = '{ip}-{ua}'.format(ip=request.remote_addr,
                                           ua=self._get_user_agent())
            alg = hashlib.md5()
            alg.update(user_hash.encode('utf8'))
            return alg.hexdigest()
        return None
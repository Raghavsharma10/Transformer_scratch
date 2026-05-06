def _makeKey(self, usern):
        """
        Make a new, probably unique key. This key will be sent in an email to
        the user and is used to access the password change form.
        """
        return unicode(hashlib.md5(str((usern, time.time(), random.random()))).hexdigest())
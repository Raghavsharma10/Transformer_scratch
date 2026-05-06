def _user_agent_digest_hash(self):
        """
        Hash the user agent and user agent password
        Section 3.10 of https://www.nar.realtor/retsorg.nsf/retsproto1.7d6.pdf
        :return: md5
        """
        if not self.version:
            raise MissingVersion("A version is required for user agent auth. The RETS server should set this"
                                 "automatically but it has not. Please instantiate the session with a version argument"
                                 "to provide the version.")
        version_number = self.version.strip('RETS/')
        user_str = '{0!s}:{1!s}'.format(self.user_agent, self.user_agent_password).encode('utf-8')
        a1 = hashlib.md5(user_str).hexdigest()
        session_id = self.session_id if self.session_id is not None else ''
        digest_str = '{0!s}::{1!s}:{2!s}'.format(a1, session_id, version_number).encode('utf-8')
        digest = hashlib.md5(digest_str).hexdigest()
        return digest
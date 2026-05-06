def delete_ace(self, domain=None, user=None, sid=None):
        """ delete ACE for the share

        delete ACE for the share.  User could either supply the domain and
        username or the sid of the user.

        :param domain: domain of the user
        :param user: username
        :param sid: sid of the user or sid list of the user
        :return: REST API response
        """
        if sid is None:
            if domain is None:
                domain = self.cifs_server.domain

            sid = UnityAclUser.get_sid(self._cli, user=user, domain=domain)
        if isinstance(sid, six.string_types):
            sid = [sid]
        ace_list = [self._make_remove_ace_entry(s) for s in sid]

        resp = self.action("setACEs", cifsShareACEs=ace_list)
        resp.raise_if_err()
        return resp
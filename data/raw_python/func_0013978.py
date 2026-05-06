def clear_access(self, white_list=None):
        """ clear all ace entries of the share

        :param white_list: list of username whose access entry won't be cleared
        :return: sid list of ace entries removed successfully
        """
        access_entries = self.get_ace_list()
        sid_list = access_entries.sid_list

        if white_list:
            sid_white_list = [UnityAclUser.get_sid(self._cli,
                                                   user,
                                                   self.cifs_server.domain)
                              for user in white_list]
            sid_list = list(set(sid_list) - set(sid_white_list))

        resp = self.delete_ace(sid=sid_list)
        resp.raise_if_err()
        return sid_list
def get_user_by_username(self, username):
        """
        Returns details for user of the given username.

        If there is more than one match will only return the first.  Use
        get_users() for full result set.
        """
        results = self.get_users(filter='username eq "%s"' % (username))
        if results['totalResults'] == 0:
            logging.warning("Found no matches for given username.")
            return
        elif results['totalResults'] > 1:
            logging.warning("Found %s matches for username %s" %
                (results['totalResults'], username))

        return results['resources'][0]
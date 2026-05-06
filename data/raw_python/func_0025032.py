def get_user_by_email(self, email):
        """
        Returns details for user with the given email address.

        If there is more than one match will only return the first.  Use
        get_users() for full result set.
        """
        results = self.get_users(filter='email eq "%s"' % (email))
        if results['totalResults'] == 0:
            logging.warning("Found no matches for given email.")
            return
        elif results['totalResults'] > 1:
            logging.warning("Found %s matches for email %s" %
                (results['totalResults'], email))

        return results['resources'][0]
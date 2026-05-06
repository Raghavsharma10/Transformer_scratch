def update_user(self, name=None, email=None, blog=None,
                    company=None, location=None, hireable=False, bio=None):
        """If authenticated as this user, update the information with
        the information provided in the parameters. All parameters are
        optional.

        :param str name: e.g., 'John Smith', not login name
        :param str email: e.g., 'john.smith@example.com'
        :param str blog: e.g., 'http://www.example.com/jsmith/blog'
        :param str company: company name
        :param str location: where you are located
        :param bool hireable: defaults to False
        :param str bio: GitHub flavored markdown
        :returns: bool
        """
        user = self.user()
        return user.update(name, email, blog, company, location, hireable,
                           bio)
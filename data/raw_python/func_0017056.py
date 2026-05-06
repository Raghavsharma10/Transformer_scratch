def update(self, name=None, email=None, blog=None, company=None,
               location=None, hireable=False, bio=None):
        """If authenticated as this user, update the information with
        the information provided in the parameters.

        :param str name: e.g., 'John Smith', not login name
        :param str email: e.g., 'john.smith@example.com'
        :param str blog: e.g., 'http://www.example.com/jsmith/blog'
        :param str company:
        :param str location:
        :param bool hireable: defaults to False
        :param str bio: GitHub flavored markdown
        :returns: bool
        """
        user = {'name': name, 'email': email, 'blog': blog,
                'company': company, 'location': location,
                'hireable': hireable, 'bio': bio}
        self._remove_none(user)
        url = self._build_url('user')
        json = self._json(self._patch(url, data=dumps(user)), 200)
        if json:
            self._update_(json)
            return True
        return False
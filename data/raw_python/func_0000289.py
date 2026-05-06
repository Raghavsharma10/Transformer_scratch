def comments(self, issue):
        """Return all comments for this issue/pull request
        """
        commit = self.as_id(issue)
        return self.get_list(url='%s/%s/comments' % (self, commit))
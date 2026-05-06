def get_comments(self):
        """
            Obtain comments for this bug.

            Returns a list of Comment instances.
        """
        bug = str(self._bug['id'])
        res = self._bugsy.request('bug/%s/comment' % bug)

        return [Comment(bugsy=self._bugsy, **comments) for comments
                in res['bugs'][bug]['comments']]
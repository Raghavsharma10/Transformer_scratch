def add_comment(self, comment):
        """
            Adds a comment to a bug. If the bug object does not have a bug ID
            (ie you are creating a bug) then you will need to also call `put`
            on the :class:`Bugsy` class.

            >>> bug.add_comment("I like sausages")
            >>> bugzilla.put(bug)

            If it does have a bug id then this will immediately post to the server

            >>> bug.add_comment("I like eggs too")

            More examples can be found at:
            https://github.com/AutomatedTester/Bugsy/blob/master/example/add_comments.py
        """
        # If we have a key post immediately otherwise hold onto it until
        # put(bug) is called
        if 'id' in self._bug:
            self._bugsy.request('bug/{}/comment'.format(self._bug['id']),
                                method='POST', json={"comment": comment}
                                )
        else:
            self._bug['comment'] = comment
def put(self, bug):
        """
            This method allows you to create or update a bug on Bugzilla. You
            will have had to pass in a valid username and password to the
            object initialisation and recieved back a token.

            :param bug: A Bug object either created by hand or by using get()

            If there is no valid token then a BugsyException will be raised.
            If the object passed in is not a Bug then a BugsyException will
            be raised.

            >>> bugzilla = Bugsy()
            >>> bug = bugzilla.get(123456)
            >>> bug.summary = "I like cheese and sausages"
            >>> bugzilla.put(bug)

        """
        if not self._have_auth:
            raise BugsyException("Unfortunately you can't put bugs in Bugzilla"
                                 " without credentials")

        if not isinstance(bug, Bug):
            raise BugsyException("Please pass in a Bug object when posting"
                                 " to Bugzilla")

        if not bug.id:
            result = self.request('bug', 'POST', json=bug.to_dict())
            if 'error' not in result:
                bug._bug['id'] = result['id']
                bug._bugsy = self
                try:
                    bug._bug.pop('comment')
                except Exception:
                    # If we don't have a `comment` we will error so let's just
                    # swallow it.
                    pass
            else:
                raise BugsyException(result['message'])
        else:
            result = self.request('bug/%s' % bug.id, 'PUT',
                                  json=bug.to_dict())
            updated_bug = self.get(bug.id)
            return updated_bug
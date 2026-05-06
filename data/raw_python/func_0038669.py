def put_bug(self, bugid, bug_update):
        '''http://bugzilla.readthedocs.org/en/latest/api/core/v1/bug.html#update-bug'''
        assert type(bug_update) is DotDict
        if (not 'ids' in bug_update):
            bug_update.ids = [bugid]

        return self._put('bug/{bugid}'.format(bugid=bugid),
                json.dumps(bug_update))
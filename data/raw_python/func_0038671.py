def post_bug(self, bug):
        '''http://bugzilla.readthedocs.org/en/latest/api/core/v1/bug.html#create-bug'''
        assert type(bug) is DotDict
        assert 'product' in bug
        assert 'component' in bug
        assert 'summary' in bug
        if (not 'version' in bug): bug.version = 'other'
        if (not 'op_sys' in bug): bug.op_sys = 'All'
        if (not 'platform' in bug): bug.platform = 'All'

        return self._post('bug', json.dumps(bug))
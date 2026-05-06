def post_comment(self, bugid, comment):
        '''http://bugzilla.readthedocs.org/en/latest/api/core/v1/comment.html#create-comments'''
        data = {'id': bugid, "comment": comment}
        return self._post('bug/{bugid}/comment'.format(bugid=bugid), json.dumps(data))
def post_attachment(self, bugid, attachment):
        '''http://bugzilla.readthedocs.org/en/latest/api/core/v1/attachment.html#create-attachment'''
        assert type(attachment) is DotDict
        assert 'data' in attachment
        assert 'file_name' in attachment
        assert 'summary' in attachment
        if (not 'content_type' in attachment): attachment.content_type = 'text/plain'
        attachment.ids = bugid
        attachment.data = base64.standard_b64encode(bytearray(attachment.data, 'ascii')).decode('ascii')

        return self._post('bug/{bugid}/attachment'.format(bugid=bugid), json.dumps(attachment))
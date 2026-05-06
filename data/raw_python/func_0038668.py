def put_attachment(self, attachmentid, attachment_update):
        '''http://bugzilla.readthedocs.org/en/latest/api/core/v1/attachment.html#update-attachment'''
        assert type(attachment_update) is DotDict
        if (not 'ids' in attachment_update):
            attachment_update.ids = [attachmentid]

        return self._put('bug/attachment/{attachmentid}'.format(attachmentid=attachmentid),
                json.dumps(attachment_update))
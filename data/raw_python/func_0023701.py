def create(self):
        """
        Submit action on the Indicator object

        :return: Indicator Object
        """
        uri = '/users/{0}/feeds/{1}/indicators'\
            .format(self.user, self.feed)

        data = {
            "indicator": json.loads(str(self.indicator)),
            "comment": self.comment,
            "content": self.content
        }

        if self.attachment:
            attachment = self._file_to_attachment(
                self.attachment, filename=self.attachment_name)

            data['attachment'] = {
                'data': attachment['data'],
                'filename': attachment['filename']
            }

        if not data['indicator'].get('indicator'):
            data['indicator']['indicator'] = attachment['sha1']

        if not data['indicator'].get('indicator'):
            raise Exception('Missing indicator')

        return self.client.post(uri, data)
def send(self):
        """Sends the broadcast message.

        :returns: tuple of (:class:`adnpy.models.Message`, :class:`adnpy.models.APIMeta`)

        """
        parse_links = self.parse_links or self.parse_markdown_links

        message = {
            'annotations': [],
            'entities': {
                'parse_links': parse_links,
                'parse_markdown_links': self.parse_markdown_links,
            }
        }

        if self.photo:
            photo, photo_meta = _upload_file(self.api, self.photo)
            message['annotations'].append({
                'type': 'net.app.core.oembed',
                'value': {
                    '+net.app.core.file': {
                        'file_id': photo.id,
                        'file_token': photo.file_token,
                        'format': 'oembed',
                    }
                }
            })

        if self.attachment:
            attachment, attachment_meta = _upload_file(self.api, self.attachment)
            message['annotations'].append({
                'type': 'net.app.core.attachments',
                'value': {
                    '+net.app.core.file_list': [
                        {
                            'file_id': attachment.id,
                            'file_token': attachment.file_token,
                            'format': 'metadata',
                        }
                    ]
                }
            })

        if self.text:
            message['text'] = self.text
        else:
            message['machine_only'] = True

        if self.headline:
            message['annotations'].append({
                'type': 'net.app.core.broadcast.message.metadata',
                'value': {
                    'subject': self.headline,
                },
            })

        if self.read_more_link:
            message['annotations'].append({
                'type': 'net.app.core.crosspost',
                'value': {
                    'canonical_url': self.read_more_link,
                }
            })

        return self.api.create_message(self.channel_id, data=message)
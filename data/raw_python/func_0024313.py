def serialize(self, user=None):
        """
        Serializes message for given user.

        Note:
            Should be called before first save(). Otherwise "is_update" will get wrong value.

        Args:
            user: User object

        Returns:
            Dict. JSON serialization ready dictionary object
        """
        return {
            'content': self.body,
            'type': self.typ,
            'updated_at': self.updated_at,
            'timestamp': self.updated_at,
            'is_update': not hasattr(self, 'unsaved'),
            'attachments': [attachment.serialize() for attachment in self.attachment_set],
            'title': self.msg_title,
            'url': self.url,
            'sender_name': self.sender.full_name,
            'sender_key': self.sender.key,
            'channel_key': self.channel.key,
            'cmd': 'message',
            'avatar_url': self.sender.avatar,
            'key': self.key,
        }
def get_channel_listing(self):
        """
        serialized form for channel listing

        """
        return {'name': self.name,
                'key': self.channel.key,
                'type': self.channel.typ,
                'read_only': self.read_only,
                'is_online': self.is_online(),
                'actions': self.get_actions(),
                'unread': self.unread_count()}
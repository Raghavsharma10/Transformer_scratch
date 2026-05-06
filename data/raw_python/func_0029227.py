def channel_name_to_id(self, channel_name, force_lookup=False):
        """Helper name for getting a channel's id from its name
        """
        if force_lookup or not self.channel_name_id_map:
            channels = self.channels_list()['channels']
            self.channel_name_id_map = {channel['name']: channel['id'] for channel in channels}
        channel = channel_name.startswith('#') and channel_name[1:] or channel_name
        return self.channel_name_id_map.get(channel)
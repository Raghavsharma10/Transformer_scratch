def move_complete_channel(self):
        """
        Channels and theirs subscribers are moved
        completely to new channel or existing channel.
        """

        to_channel = Channel.objects.get(self.current.task_data['target_channel_key'])
        chosen_channels = self.current.task_data['chosen_channels']
        chosen_channels_names = self.current.task_data['chosen_channels_names']

        with BlockSave(Subscriber, query_dict={'channel_id': to_channel.key}):
            for s in Subscriber.objects.filter(channel_id__in=chosen_channels, typ=15):
                s.channel = to_channel
                s.save()

        with BlockDelete(Message):
            Message.objects.filter(channel_id__in=chosen_channels, typ=15).delete()

        with BlockDelete(Channel):
            Channel.objects.filter(key__in=chosen_channels).delete()

        self.current.task_data[
            'msg'] = _(u"Chosen channels(%s) have been merged to '%s' channel successfully.") % \
                     (', '.join(chosen_channels_names), to_channel.name)
def move_chosen_subscribers(self):
        """
        After splitting operation, only chosen subscribers
        are moved to new channel or existing channel.
        """
        from_channel = Channel.objects.get(self.current.task_data['chosen_channels'][0])
        to_channel = Channel.objects.get(self.current.task_data['target_channel_key'])

        with BlockSave(Subscriber, query_dict={'channel_id': to_channel.key}):
            for subscriber in Subscriber.objects.filter(
                    key__in=self.current.task_data['chosen_subscribers']):
                subscriber.channel = to_channel
                subscriber.save()

        if self.current.task_data['new_channel']:
            self.copy_and_move_messages(from_channel, to_channel)

        self.current.task_data[
            'msg'] = _(u"Chosen subscribers and messages of them migrated from '%s' channel to "
                       u"'%s' channel successfully.") % (from_channel.name, to_channel.name)
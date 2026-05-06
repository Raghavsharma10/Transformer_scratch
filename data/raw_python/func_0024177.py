def copy_and_move_messages(from_channel, to_channel):
        """
         While splitting channel and moving chosen subscribers to new channel,
         old channel's messages are copied and moved to new channel.

         Args:
            from_channel (Channel object): move messages from channel
            to_channel (Channel object): move messages to channel
        """
        with BlockSave(Message, query_dict={'channel_id': to_channel.key}):
            for message in Message.objects.filter(channel=from_channel, typ=15):
                message.key = ''
                message.channel = to_channel
                message.save()
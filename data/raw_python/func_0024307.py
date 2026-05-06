def get_or_create_direct_channel(cls, initiator_key, receiver_key):
        """
        Creates a  direct messaging channel between two user

        Args:
            initiator: User, who want's to make first contact
            receiver: User, other party

        Returns:
            (Channel, receiver_name)
        """
        existing = cls.objects.OR().filter(
            code_name='%s_%s' % (initiator_key, receiver_key)).filter(
            code_name='%s_%s' % (receiver_key, initiator_key))
        receiver_name = UserModel.objects.get(receiver_key).full_name
        if existing:
            channel = existing[0]
        else:
            channel_name = '%s_%s' % (initiator_key, receiver_key)
            channel = cls(is_direct=True, code_name=channel_name, typ=10).blocking_save()
        with BlockSave(Subscriber):
            Subscriber.objects.get_or_create(channel=channel,
                                             user_id=initiator_key,
                                             name=receiver_name)
            Subscriber.objects.get_or_create(channel=channel,
                                             user_id=receiver_key,
                                             name=UserModel.objects.get(initiator_key).full_name)
        return channel, receiver_name
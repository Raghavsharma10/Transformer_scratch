def publish_participation_to_thread(sender, instance, created, **kwargs):
    """ Warns users everytime a thread including them is published. This is done via channel subscription.  """
    if kwargs.get('created_and_add_participants') is True:
        request_participant_id = kwargs.get('request_participant_id')
        if request_participant_id is not None:
            client = Client("{0}api/".format(getattr(settings, "CENTRIFUGE_ADDRESS")), getattr(settings, "CENTRIFUGE_SECRET"))
            active_participants = [participation.participant for participation in Participation.objects.filter(thread=instance, date_left__isnull=True).select_related('participant')]
            for participant in active_participants:
                client.publish(
                    build_channel(settings.CENTRIFUGO_THREAD_NAMESPACE, participant.id, [participant.id]),
                    {
                        "message_channel_to_connect_to": build_channel(settings.CENTRIFUGO_MESSAGE_NAMESPACE, instance.id, [p.id for p in active_participants])
                    }
                )
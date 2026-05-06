def publish_message_to_centrifugo(sender, instance, created, **kwargs):
    """ Publishes each saved message to Centrifugo. """
    if created is True:
        client = Client("{0}api/".format(getattr(settings, "CENTRIFUGE_ADDRESS")), getattr(settings, "CENTRIFUGE_SECRET"))
        # we ensure the client is still in the thread (he may have left or have been removed)
        active_participants = [participation.participant.id for participation in Participation.objects.filter(thread=instance.thread, date_left__isnull=True).select_related('participant')]
        client.publish(
            build_channel(settings.CENTRIFUGO_MESSAGE_NAMESPACE, instance.thread.id, active_participants),
            {
                "id": instance.id,
                "body": instance.body,
                "sender": instance.sender.id,
                "thread": instance.thread.id,
                "sent_at": str(instance.sent_at),
                "is_notification": True,  # ATTENTION: check against sender too to be sure to not notify him his message
            }
        )
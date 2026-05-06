def post(self, request, *args, **kwargs):
        """
        Returns a token identifying the user in Centrifugo.
        """

        current_timestamp = "%.0f" % time.time()
        user_id_str = u"{0}".format(request.user.id)
        token = generate_token(settings.CENTRIFUGE_SECRET, user_id_str, "{0}".format(current_timestamp), info="")

        # we get all the channels to which the user can subscribe
        participant = Participant.objects.get(id=request.user.id)

        # we use the threads as channels ids
        channels = []
        for thread in Thread.managers.get_threads_where_participant_is_active(participant_id=participant.id):
            channels.append(
                build_channel(settings.CENTRIFUGO_MESSAGE_NAMESPACE, thread.id, thread.participants.all())
            )

        # we also have a channel to alert us about new threads
        threads_channel = build_channel(settings.CENTRIFUGO_THREAD_NAMESPACE, request.user.id, [request.user.id])  # he is the only one to have access to the channel
        channels.append(threads_channel)

        # we return the information
        to_return = {
            'user': user_id_str,
            'timestamp': current_timestamp,
            'token': token,
            'connection_url': "{0}connection/".format(settings.CENTRIFUGE_ADDRESS),
            'channels': channels,
            'debug': settings.DEBUG,
        }

        return HttpResponse(json.dumps(to_return), content_type='application/json; charset=utf-8')
def get_or_create_thread(self, request, name=None, *participant_ids):
        """
        When a Participant posts a message to other participants without specifying an existing Thread,
        we must
        1. Create a new Thread if they have not yet opened the discussion.
        2. If they have already opened the discussion and multiple Threads are not allowed for the same users, we must
            re-attach this message to the existing thread.
        3. If they have already opened the discussion and multiple Threads are allowed, we simply create a new one.
        """

        # we get the current participant
        # or create him if he does not exit

        participant_ids = list(participant_ids)
        if request.rest_messaging_participant.id not in participant_ids:
            participant_ids.append(request.rest_messaging_participant.id)

        # we need at least one other participant
        if len(participant_ids) < 2:
            raise Exception('At least two participants are required.')

        if getattr(settings, "REST_MESSAGING_THREAD_UNIQUE_FOR_ACTIVE_RECIPIENTS", True) is True:
            # if we limit the number of threads by active participants
            # we ensure a thread is not already running
            existing_threads = self.get_active_threads_involving_all_participants(*participant_ids)
            if len(list(existing_threads)) > 0:
                return existing_threads[0]

        # we have no existing Thread or multiple Thread instances are allowed
        thread = Thread.objects.create(name=name)

        # we add the participants
        thread.add_participants(request, *participant_ids)

        # we send a signal to say the thread with participants is created
        post_save.send(Thread, instance=thread, created=True, created_and_add_participants=True, request_participant_id=request.rest_messaging_participant.id)

        return thread
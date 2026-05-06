def get_all_messages_in_thread(self, participant_id, thread_id, check_who_read=True):
        """ Returns all the messages in a thread. """
        try:
            messages = Message.objects.filter(thread__id=thread_id).\
                order_by('-id').\
                select_related('thread').\
                prefetch_related('thread__participation_set', 'thread__participation_set__participant')
        except Exception:
            return Message.objects.none()

        messages = self.check_who_read(messages)
        return messages
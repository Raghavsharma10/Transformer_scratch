def get_lasts_messages_of_threads(self, participant_id, check_who_read=True, check_is_notification=True):
        """ Returns the last message in each thread """
        # we get the last message for each thread
        # we must query the messages using two queries because only Postgres supports .order_by('thread', '-sent_at').distinct('thread')
        threads = Thread.managers.\
            get_threads_where_participant_is_active(participant_id).\
            annotate(last_message_id=Max('message__id'))
        messages = Message.objects.filter(id__in=[thread.last_message_id for thread in threads]).\
            order_by('-id').\
            distinct().\
            select_related('thread', 'sender')

        if check_who_read is True:
            messages = messages.prefetch_related('thread__participation_set', 'thread__participation_set__participant')
            messages = self.check_who_read(messages)
        else:
            messages = messages.prefetch_related('thread__participants')

        if check_is_notification is True:
            messages = self.check_is_notification(participant_id, messages)
        return messages
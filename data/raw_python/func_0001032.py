def get_threads_where_participant_is_active(self, participant_id):
        """ Gets all the threads in which the current participant is involved. The method excludes threads where the participant has left. """
        participations = Participation.objects.\
            filter(participant__id=participant_id).\
            exclude(date_left__lte=now()).\
            distinct().\
            select_related('thread')

        return Thread.objects.\
            filter(id__in=[p.thread.id for p in participations]).\
            distinct()
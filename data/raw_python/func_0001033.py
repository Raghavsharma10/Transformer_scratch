def get_active_threads_involving_all_participants(self, *participant_ids):
        """ Gets the threads where the specified participants are active and no one has left. """

        query = Thread.objects.\
            exclude(participation__date_left__lte=now()).\
            annotate(count_participants=Count('participants')).\
            filter(count_participants=len(participant_ids))

        for participant_id in participant_ids:
            query = query.filter(participants__id=participant_id)

        return query.distinct()
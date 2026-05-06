def check_is_notification(self, participant_id, messages):
        """ Check if each message requires a notification for the specified participant. """
        try:
            # we get the last check
            last_check = NotificationCheck.objects.filter(participant__id=participant_id).latest('id').date_check
        except Exception:
            # we have no notification check
            # all the messages are considered as new
            for m in messages:
                m.is_notification = True
            return messages

        for m in messages:
            if m.sent_at > last_check and m.sender.id != participant_id:
                setattr(m, "is_notification", True)
            else:
                setattr(m, "is_notification", False)
        return messages
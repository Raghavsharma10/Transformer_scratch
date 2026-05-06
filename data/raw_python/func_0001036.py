def check_who_read(self, messages):
        """ Check who read each message. """
        # we get the corresponding Participation objects
        for m in messages:
            readers = []
            for p in m.thread.participation_set.all():
                if p.date_last_check is None:
                    pass
                elif p.date_last_check > m.sent_at:
                    # the message has been read
                    readers.append(p.participant.id)
            setattr(m, "readers", readers)

        return messages
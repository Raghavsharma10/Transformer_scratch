def add_answer_at_time(self, record, now):
        """Adds an answer if if does not expire by a certain time"""
        if record is not None:
            if now == 0 or not record.is_expired(now):
                self.answers.append((record, now))
                if record.rrsig is not None:
                    self.answers.append((record.rrsig, now))
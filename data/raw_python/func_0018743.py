def add_listener(self, listener, question):
        """Adds a listener for a given question.  The listener will have
        its update_record method called when information is available to
        answer the question."""
        now = current_time_millis()
        self.listeners.append(listener)
        if question is not None:
            for record in self.cache.entries_with_name(question.name):
                if question.answered_by(record) and not record.is_expired(now):
                    listener.update_record(self, now, record)
        self.notify_all()
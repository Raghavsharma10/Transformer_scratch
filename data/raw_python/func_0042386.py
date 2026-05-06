def pop_message(self, wait=SECOND, till=None):
        """
        RETURN TUPLE (message, payload) CALLER IS RESPONSIBLE FOR CALLING message.delete() WHEN DONE
        """
        if till is not None and not isinstance(till, Signal):
            Log.error("Expecting a signal")

        message = self.queue.read(wait_time_seconds=mo_math.floor(wait.seconds))
        if not message:
            return None
        message.delete = lambda: self.queue.delete_message(message)

        payload = mo_json.json2value(message.get_body())
        return message, payload
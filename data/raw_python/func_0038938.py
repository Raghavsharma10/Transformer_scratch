def send_message(self, id_service, content):
        """Write all messages to the stream in a thread-safe way."""
        if not content:
            return
        with self._lock:
            try:
                message = "Message: %s to %s" % (content, id_service)
                self.write_message(message)
                self.stream.flush()  # flush after each message
                return "message_id"
            except Exception:
                if not self.fail_silently:
                    raise
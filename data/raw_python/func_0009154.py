def send_message(self, msg_dict):
        """Serialize a message dictionary and write it to the output stream."""
        with self._writer_lock:
            try:
                self.output_stream.flush()
                self.output_stream.write(self.serialize_dict(msg_dict))
                self.output_stream.flush()
            except IOError:
                raise StormWentAwayError()
            except:
                log.exception("Failed to send message: %r", msg_dict)
def data_received(self, data):
        """Handle data received."""
        self.tokenizer.feed(data)
        while self.tokenizer.has_tokens():
            raw = self.tokenizer.get_next_token()
            frame = frame_from_raw(raw)
            if frame is not None:
                self.frame_received_cb(frame)
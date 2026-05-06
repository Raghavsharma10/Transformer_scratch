def on_key_up(self):
        """ Process key up event by updating buffer and release key. """
        if (self.last_pressed is not None):
            self.set_key_state(self.last_pressed, 0)
            self.buffer = self.last_pressed.update_buffer(self.buffer)
            self.text_consumer(self.buffer)
            self.last_pressed = None
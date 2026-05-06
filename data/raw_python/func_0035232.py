def set_volume(self, volume):
        """
        Sets player volume (note, this does not change host computer main volume).
        """
        msg = cr.Message()
        msg.type = cr.SET_VOLUME
        msg.request_set_volume.volume = int(volume)
        self.send_message(msg)
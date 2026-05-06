def from_payload(self, payload):
        """Init frame from binary data."""
        number_of_objects = payload[0]
        self.remaining_scenes = payload[-1]
        predicted_len = number_of_objects * 65 + 2
        if len(payload) != predicted_len:
            raise PyVLXException('scene_list_notification_wrong_length')
        self.scenes = []
        for i in range(number_of_objects):
            scene = payload[(i*65+1):(i*65+66)]
            number = scene[0]
            name = bytes_to_string(scene[1:])
            self.scenes.append((number, name))
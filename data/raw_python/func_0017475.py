def poll_for_response(self):
        """
        Polls the device for user input

        If there is a keymapping for the device, the key map is applied
        to the key reported from the device.

        If a response is waiting to be processed, the response is appended
        to the internal response_queue
        """
        key_state = self.con.check_for_keypress()

        if key_state != NO_KEY_DETECTED:
            response = self.con.get_current_response()

            if self.keymap is not None:
                response['key'] = self.keymap[response['key']]
            else:
                response['key'] -= 1

            self.response_queue.append(response)
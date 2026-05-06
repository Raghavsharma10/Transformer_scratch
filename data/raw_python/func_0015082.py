def _parse(self, msg):
        """
        Parses a Scratch message and returns a tuple with the first element
        as the message type, and the second element as the message payload. The 
        payload for a 'broadcast' message is a string, and the payload for a 
        'sensor-update' message is a dict whose keys are variables, and values
        are updated variable values. Returns None if msg is not a message.
        """
        if not self._is_msg(msg):
            return None
        msg_type = msg[self.prefix_len:].split(' ')[0]
        if msg_type == 'broadcast':
            return ('broadcast', self._parse_broadcast(msg))
        else:
            return ('sensor-update', self._parse_sensorupdate(msg))
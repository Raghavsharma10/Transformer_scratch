def get_roaster_state(self):
        """Returns a string based upon the current state of the roaster. Will
        raise an exception if the state is unknown.

        Returns:
            'idle' if idle,
            'sleeping' if sleeping,
            'cooling' if cooling,
            'roasting' if roasting,
            'connecting' if in hardware connection phase,
            'unknown' otherwise
        """
        value = self._current_state.value
        if(value == b'\x02\x01'):
            return 'idle'
        elif(value == b'\x04\x04'):
            return 'cooling'
        elif(value == b'\x08\x01'):
            return 'sleeping'
        # handle null bytes as empty strings
        elif(value == b'\x00\x00' or value == b''):
            return 'connecting'
        elif(value == b'\x04\x02'):
            return 'roasting'
        else:
            return 'unknown'
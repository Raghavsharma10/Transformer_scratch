def system_status(self):
        """The system status codes."""
        flag, timestamp, status = self._query(('GETDAT? 1', (Integer, Float, Integer)))
        return {
            # convert unix timestamp to datetime object
            'timestamp': datetime.datetime.fromtimestamp(timestamp),
            # bit 0-3 represent the temperature controller status
            'temperature': STATUS_TEMPERATURE[status & 0xf],
            # bit 4-7 represent the magnet status
            'magnet': STATUS_MAGNET[(status >> 4) & 0xf],
            # bit 8-11 represent the chamber status
            'chamber': STATUS_CHAMBER[(status >> 8) & 0xf],
            # bit 12-15 represent the sample position status
            'sample_position': STATUS_SAMPLE_POSITION[(status >> 12) & 0xf],
        }
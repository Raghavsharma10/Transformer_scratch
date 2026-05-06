def handle_set_key(self):
        """Read incoming key from server"""
        track_id = self.reader.int()
        row = self.reader.int()
        value = self.reader.float()
        kind = self.reader.byte()
        logger.info(" -> track=%s, row=%s, value=%s, type=%s", track_id, row, value, kind)

        # Add or update track value
        track = self.tracks.get_by_id(track_id)
        track.add_or_update(row, value, kind)
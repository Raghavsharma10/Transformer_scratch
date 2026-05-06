def handle_delete_key(self):
        """Read incoming delete key event from server"""
        track_id = self.reader.int()
        row = self.reader.int()
        logger.info(" -> track=%s, row=%s", track_id, row)

        # Delete the actual track value
        track = self.tracks.get_by_id(track_id)
        track.delete(row)
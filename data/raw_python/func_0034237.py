def get_key_at(self, position):
        """Retrieves if any key is located at the given position
        
        :param position: Position to check key at.
        :returns: The located key if any at the given position, None otherwise.
        """
        for row in self.rows:
            if position in row:
                for key in row.keys:
                    if key.is_touched(position):
                        return key
        return None
def delete(self, row):
        """Delete a track value"""
        i = self._get_key_index(row)
        del self.keys[i]
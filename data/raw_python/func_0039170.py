def value(self, name):
        """get value of a track at the current time"""
        return self.tracks.get(name).row_value(self.controller.row)
def add(self, obj):
        """
        Add pre-created tracks.
        If the tracks are already created, we hijack the data.
        This way the pointer to the pre-created tracks are still valid.
        """
        obj.controller = self.controller
        # Is the track already loaded or created?
        track = self.tracks.get(obj.name)
        if track:
            if track == obj:
                return
            # hijack the track data
            obj.keys = track.keys
            obj.controller = track.controller
            self.tracks[track.name] = obj
            self.track_index[self.track_index.index(track)] = obj
        else:
            # Add a new track
            obj.controller = self.controller
            self.tracks[obj.name] = obj
            self.track_index.append(obj)
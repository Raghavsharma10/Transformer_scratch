def empty_over_span(self, time, duration):
        """Helper method that tests whether composition contains any segments
        at a given time for a given duration. 

        :param time: Time (in seconds) to start span
        :param duration: Duration (in seconds) of span
        :returns: `True` if there are no segments in the composition that overlap the span starting at `time` and lasting for `duration` seconds. `False` otherwise.
        """
        for seg in self.segments:
            # starts in range
            if seg.comp_location_in_seconds >= time and\
                seg.comp_location_in_seconds < time + duration:
                return False
            # or, ends in range
            elif seg.comp_location_in_seconds + seg.duration_in_seconds >= time and\
                seg.comp_location_in_seconds + seg.duration_in_seconds < time + duration:
                return False
            # or, spans entire range
            elif seg.comp_location_in_seconds < time and\
                seg.comp_location_in_seconds + seg.duration_in_seconds >= time + duration:
                return False
        return True
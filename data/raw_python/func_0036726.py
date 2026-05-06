def fade_out(self, segment, duration, fade_type="linear"):
        """Adds a fade out to a segment in the composition

        :param segment: Segment to fade out
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-out (in seconds)
        :type duration: float
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        score_loc_in_seconds = segment.comp_location_in_seconds +\
            segment.duration_in_seconds - duration

        f = Fade(segment.track, score_loc_in_seconds, duration, 1.0, 0.0,
                 fade_type=fade_type)
        # bug fixing... perhaps
        f.comp_location = segment.comp_location + segment.duration -\
            int(duration * segment.track.samplerate)
        self.add_dynamic(f)
        return f
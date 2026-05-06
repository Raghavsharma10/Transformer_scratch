def extended_fade_out(self, segment, duration):
        """Add a fade-out to a segment that extends the beginning of the
        segment.

        :param segment: Segment to fade out
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-out (in seconds)
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        dur = int(duration * segment.track.samplerate)
        if segment.start + segment.duration + dur <\
                segment.track.duration:
            segment.duration += dur
        else:
            raise Exception(
                "Cannot create fade-out that extends past the track's end")
        score_loc_in_seconds = segment.comp_location_in_seconds +\
            segment.duration_in_seconds - duration
        f = Fade(segment.track, score_loc_in_seconds, duration, 1.0, 0.0)
        self.add_dynamic(f)
        return f
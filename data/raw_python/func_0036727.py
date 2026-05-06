def extended_fade_in(self, segment, duration):
        """Add a fade-in to a segment that extends the beginning of the
        segment.

        :param segment: Segment to fade in
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-in (in seconds)
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        dur = int(duration * segment.track.samplerate)
        if segment.start - dur >= 0:
            segment.start -= dur
        else:
            raise Exception(
                "Cannot create fade-in that extends "
                "past the track's beginning")
        if segment.comp_location - dur >= 0:
            segment.comp_location -= dur
        else:
            raise Exception(
                "Cannot create fade-in the extends past the score's beginning")

        segment.duration += dur

        f = Fade(segment.track, segment.comp_location_in_seconds,
                 duration, 0.0, 1.0)
        self.add_dynamic(f)
        return f
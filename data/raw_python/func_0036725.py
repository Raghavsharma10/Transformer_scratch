def fade_in(self, segment, duration, fade_type="linear"):
        """Adds a fade in to a segment in the composition

        :param segment: Segment to fade in to
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-in (in seconds)
        :type duration: float
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        f = Fade(segment.track, segment.comp_location_in_seconds,
                 duration, 0.0, 1.0, fade_type=fade_type)
        self.add_dynamic(f)
        return f
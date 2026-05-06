def add_segments(self, segments):
        """Add a list of segments to the composition

        :param segments: Segments to add to composition
        :type segments: list of :py:class:`radiotool.composer.Segment`
        """
        self.tracks.update([seg.track for seg in segments])
        self.segments.extend(segments)
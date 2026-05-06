def add_segment(self, segment):
        """Add a segment to the composition

        :param segment: Segment to add to composition
        :type segment: :py:class:`radiotool.composer.Segment`
        """
        self.tracks.add(segment.track)
        self.segments.append(segment)
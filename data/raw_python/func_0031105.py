def add(self, start, end):
        """
        Add the start and end offsets of a matching read.

        @param start: The C{int} start offset of the read match in the subject.
        @param end: The C{int} end offset of the read match in the subject.
            This is Python-style: the end offset is not included in the match.
        """
        assert start <= end
        self._intervals.append((start, end))
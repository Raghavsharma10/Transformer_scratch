def collect_segment_partitions(self):
        """Return a dict of segments partitions, keyed on the name of the parent partition
        """
        from collections import defaultdict

        # Group the segments by their parent partition name, which is the
        # same name, but without the segment.
        partitions = defaultdict(set)
        for p in self.dataset.partitions:
            if p.type == p.TYPE.SEGMENT:
                name = p.identity.name
                name.segment = None
                partitions[name].add(p)

        return partitions
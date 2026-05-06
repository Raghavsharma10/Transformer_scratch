def unify_partitions(self):
        """For all of the segments for a partition, create the parent partition, combine the
        children into the parent, and delete the children. """

        partitions = self.collect_segment_partitions()

        # For each group, copy the segment partitions to the parent partitions, then
        # delete the segment partitions.

        with self.progress.start('coalesce', 0, message='Coalescing partition segments') as ps:

            for name, segments in iteritems(partitions):
                ps.add(item_type='partitions', item_count=len(segments),
                       message='Colescing partition {}'.format(name))
                self.unify_partition(name, segments, ps)
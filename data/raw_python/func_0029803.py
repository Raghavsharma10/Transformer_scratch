def next_sequence_id(self, table_class, force_query=False):
        """Return the next sequence id for a object, identified by the vid of the parent object, and the database prefix
        for the child object. On the first call, will load the max sequence number
        from the database, but subsequence calls will run in process, so this isn't suitable for
        multi-process operation -- all of the tables in a dataset should be created by one process

        The child table must have a sequence_id value.

        """

        from . import next_sequence_id
        from sqlalchemy.orm import object_session

        # NOTE: This next_sequence_id uses a different algorithm than dataset.next_sequence_id
        # FIXME replace this one with dataset.next_sequence_id
        return next_sequence_id(object_session(self), self._sequence_ids, self.vid, table_class, force_query=force_query)
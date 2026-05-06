def emit(
        self, tup, tup_id=None, stream=None, direct_task=None, need_task_ids=False
    ):
        """Emit a spout Tuple & add metadata about it to `unacked_tuples`.

        In order for this to work, `tup_id` is a required parameter.

        See :meth:`Bolt.emit`.
        """
        if tup_id is None:
            raise ValueError(
                "You must provide a tuple ID when emitting with a "
                "ReliableSpout in order for the tuple to be "
                "tracked."
            )
        args = (tup, stream, direct_task, need_task_ids)
        self.unacked_tuples[tup_id] = args
        return super(ReliableSpout, self).emit(
            tup,
            tup_id=tup_id,
            stream=stream,
            direct_task=direct_task,
            need_task_ids=need_task_ids,
        )
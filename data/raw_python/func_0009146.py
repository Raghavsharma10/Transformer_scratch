def emit(self, tup, **kwargs):
        """Modified emit that will not return task IDs after emitting.

        See :class:`pystorm.component.Bolt` for more information.

        :returns: ``None``.
        """
        kwargs["need_task_ids"] = False
        return super(BatchingBolt, self).emit(tup, **kwargs)
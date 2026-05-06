def emit(
        self, tup, tup_id=None, stream=None, direct_task=None, need_task_ids=False
    ):
        """Emit a spout Tuple message.

        :param tup: the Tuple to send to Storm, should contain only
                    JSON-serializable data.
        :type tup: list or tuple
        :param tup_id: the ID for the Tuple. Leave this blank for an
                       unreliable emit.
        :type tup_id: str
        :param stream: ID of the stream this Tuple should be emitted to.
                       Leave empty to emit to the default stream.
        :type stream: str
        :param direct_task: the task to send the Tuple to if performing a
                            direct emit.
        :type direct_task: int
        :param need_task_ids: indicate whether or not you'd like the task IDs
                              the Tuple was emitted (default: ``False``).
        :type need_task_ids: bool

        :returns: ``None``, unless ``need_task_ids=True``, in which case it will
                  be a ``list`` of task IDs that the Tuple was sent to if. Note
                  that when specifying direct_task, this will be equal to
                  ``[direct_task]``.
        """
        return super(Spout, self).emit(
            tup,
            tup_id=tup_id,
            stream=stream,
            direct_task=direct_task,
            need_task_ids=need_task_ids,
        )
def emit(
        self, tup, stream=None, anchors=None, direct_task=None, need_task_ids=False
    ):
        """Emit a new Tuple to a stream.

        :param tup: the Tuple payload to send to Storm, should contain only
                    JSON-serializable data.
        :type tup: :class:`list` or :class:`pystorm.component.Tuple`
        :param stream: the ID of the stream to emit this Tuple to. Specify
                       ``None`` to emit to default stream.
        :type stream: str
        :param anchors: IDs the Tuples (or :class:`pystorm.component.Tuple`
                        instances) which the emitted Tuples should be anchored
                        to. If ``auto_anchor`` is set to ``True`` and
                        you have not specified ``anchors``, ``anchors`` will be
                        set to the incoming/most recent Tuple ID(s).
        :type anchors: list
        :param direct_task: the task to send the Tuple to.
        :type direct_task: int
        :param need_task_ids: indicate whether or not you'd like the task IDs
                              the Tuple was emitted (default: ``False``).
        :type need_task_ids: bool

        :returns: ``None``, unless ``need_task_ids=True``, in which case it will
                  be a ``list`` of task IDs that the Tuple was sent to if. Note
                  that when specifying direct_task, this will be equal to
                  ``[direct_task]``.
        """
        if anchors is None:
            anchors = self._current_tups if self.auto_anchor else []
        anchors = [a.id if isinstance(a, Tuple) else a for a in anchors]

        return super(Bolt, self).emit(
            tup,
            stream=stream,
            anchors=anchors,
            direct_task=direct_task,
            need_task_ids=need_task_ids,
        )
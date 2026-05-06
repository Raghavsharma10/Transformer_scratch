def emit(
        self,
        tup,
        tup_id=None,
        stream=None,
        anchors=None,
        direct_task=None,
        need_task_ids=False,
    ):
        """Emit a new Tuple to a stream.

        :param tup: the Tuple payload to send to Storm, should contain only
                    JSON-serializable data.
        :type tup: :class:`list` or :class:`pystorm.component.Tuple`
        :param tup_id: the ID for the Tuple. If omitted by a
                       :class:`pystorm.spout.Spout`, this emit will be
                       unreliable.
        :type tup_id: str
        :param stream: the ID of the stream to emit this Tuple to. Specify
                       ``None`` to emit to default stream.
        :type stream: str
        :param anchors: IDs the Tuples (or
                        :class:`pystorm.component.Tuple` instances)
                        which the emitted Tuples should be anchored to. This is
                        only passed by :class:`pystorm.bolt.Bolt`.
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
        if not isinstance(tup, (list, tuple)):
            raise TypeError(
                "All Tuples must be either lists or tuples, "
                "received {!r} instead.".format(type(tup))
            )

        msg = {"command": "emit", "tuple": tup}
        downstream_task_ids = None

        if anchors is not None:
            msg["anchors"] = anchors
        if tup_id is not None:
            msg["id"] = tup_id
        if stream is not None:
            msg["stream"] = stream
        if direct_task is not None:
            msg["task"] = direct_task
            if need_task_ids:
                downstream_task_ids = [direct_task]

        if not need_task_ids:
            # only need to send on False, Storm's default is True
            msg["need_task_ids"] = need_task_ids

        if need_task_ids and direct_task is None:
            # Use both locks so we ensure send_message and read_task_ids are for
            # same emit
            with self._reader_lock, self._writer_lock:
                self.send_message(msg)
                downstream_task_ids = self.read_task_ids()
        # No locks necessary in simple case because serializer will acquire
        # write lock itself
        else:
            self.send_message(msg)

        return downstream_task_ids
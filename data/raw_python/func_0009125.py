def fail(self, tup_id):
        """Called when a Tuple fails in the topology

        A reliable spout will replay a failed tuple up to ``max_fails`` times.

        :param tup_id: the ID of the Tuple that has failed in the topology
                       either due to a bolt calling ``fail()`` or a Tuple
                       timing out.
        :type tup_id: str
        """
        saved_args = self.unacked_tuples.get(tup_id)
        if saved_args is None:
            self.logger.error("Received fail for unknown tuple ID: %r", tup_id)
            return
        tup, stream, direct_task, need_task_ids = saved_args
        if self.failed_tuples[tup_id] < self.max_fails:
            self.emit(
                tup,
                tup_id=tup_id,
                stream=stream,
                direct_task=direct_task,
                need_task_ids=need_task_ids,
            )
            self.failed_tuples[tup_id] += 1
        else:
            # Just pretend we got an ack when we exceed retry limit
            self.logger.info(
                "Acking tuple ID %r after it exceeded retry limit " "(%r)",
                tup_id,
                self.max_fails,
            )
            self.ack(tup_id)
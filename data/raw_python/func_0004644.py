def build(self):
        """Create and start up the internal workers."""

        # If there's no output tube, it means that this stage
        # is at the end of a fork (hasn't been linked to any stage downstream).
        # Therefore, create one output tube.
        if not self._output_tubes:
            self._output_tubes.append(self._worker_class.getTubeClass()())

        self._worker_class.assemble(
            self._worker_args,
            self._input_tube,
            self._output_tubes,
            self._size,
            self._disable_result,
            self._do_stop_task,
            )

        # Build all downstream stages.
        for stage in self._next_stages:
            stage.build()
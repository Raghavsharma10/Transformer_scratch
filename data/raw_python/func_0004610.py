def init2(
        self, 
        input_tube,      # Read task from the input tube.
        output_tubes,    # Send result on all the output tubes.
        num_workers,     # Total number of workers in the stage.
        disable_result,  # Whether to override any result with None.
        do_stop_task,    # Whether to call doTask() on "stop" request.
        ):
        """Create *num_workers* worker objects with *input_tube* and 
        an iterable of *output_tubes*. The worker reads a task from *input_tube* 
        and writes the result to *output_tubes*."""

        super(UnorderedWorker, self).__init__()
        self._tube_task_input = input_tube
        self._tubes_result_output = output_tubes
        self._num_workers = num_workers
        self._disable_result = disable_result
        self._do_stop_task = do_stop_task
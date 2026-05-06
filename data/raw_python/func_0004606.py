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

        super(OrderedWorker, self).__init__()
        self._tube_task_input = input_tube
        self._tubes_result_output = output_tubes
        self._num_workers = num_workers

        # Serializes reading from input tube.
        self._lock_prev_input = None
        self._lock_next_input = None

        # Serializes writing to output tube.
        self._lock_prev_output = None
        self._lock_next_output = None

        self._disable_result = disable_result
        self._do_stop_task = do_stop_task
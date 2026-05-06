def assemble(
        cls, 
        args, 
        input_tube, 
        output_tubes, 
        size, 
        disable_result=False,
        do_stop_task=False,
        ):
        """Create, assemble and start workers.
        Workers are created of class *cls*, initialized with *args*, and given
        task/result communication channels *input_tube* and *output_tubes*.
        The number of workers created is according to *size* parameter.
        *do_stop_task* indicates whether doTask() will be called for "stop" request.
        """

        # Create the workers.
        workers = []
        for ii in range(size):
            worker = cls(**args)
            worker.init2(
                input_tube,
                output_tubes,
                size,
                disable_result,
                do_stop_task,
                )
            workers.append(worker)

        # Connect the workers.
        for ii in range(size):
            worker_this = workers[ii]
            worker_prev = workers[ii-1]
            worker_prev._link(
                worker_this, 
                next_is_first=(ii==0),  # Designate 0th worker as the first.
                )

        # Start the workers.
        for worker in workers:
            worker.start()
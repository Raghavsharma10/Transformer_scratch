def set_generation_type(self, num_processors=-1, num_splits=1000, verbose=-1):
        """Change generation type.

        Choose weather to generate the data in parallel or on a single processor.

        Args:
            num_processors (int or None, optional): Number of parallel processors to use.
                If ``num_processors==-1``, this will use multiprocessing module and use
                available cpus. If single generation is desired, num_processors is set
                to ``None``. Default is -1.
            num_splits (int, optional): Number of binaries to run during each process.
                Default is 1000.
            verbose (int, optional): Describes the notification of when parallel processes
                are finished. Value describes cadence of process completion notifications.
                If ``verbose == -1``, no notifications are given. Default is -1.

        """
        self.parallel_input.num_processors = num_processors
        self.parallel_input.num_splits = num_splits
        self.parallel_input.verbose = verbose
        return
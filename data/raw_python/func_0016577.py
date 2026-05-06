def max_num_reads(self, **params):
        """Returns the maximum number of reads for the given solver parameters.

        Args:
            **params:
                Parameters for the sampling method. Relevant to num_reads:

                - annealing_time
                - readout_thermalization
                - num_reads
                - programming_thermalization

        Returns:
            int: The maximum number of reads.

        """
        # dev note: in the future it would be good to have a way of doing this
        # server-side, as we are duplicating logic here.

        properties = self.properties

        if self.software or not params:
            # software solvers don't use any of the above parameters
            return properties['num_reads_range'][1]

        # qpu

        _, duration = properties['problem_run_duration_range']

        annealing_time = params.get('annealing_time',
                                    properties['default_annealing_time'])

        readout_thermalization = params.get('readout_thermalization',
                                            properties['default_readout_thermalization'])

        programming_thermalization = params.get('programming_thermalization',
                                                properties['default_programming_thermalization'])

        return min(properties['num_reads_range'][1],
                   int((duration - programming_thermalization)
                       / (annealing_time + readout_thermalization)))
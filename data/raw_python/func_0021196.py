def pause(self, queue_name, kw_in=None, kw_out=None, kw_all=None,
              kw_none=None, kw_state=None, kw_bcast=None):
        """
        Pause a queue.

        Unfortunately, the PAUSE keywords are mostly reserved words in Python,
        so I've been a little creative in the function variable names. Open
        to suggestions to change it (canardleteer)

        :param queue_name: The job queue we are modifying.
        :param kw_in: pause the queue in input.
        :param kw_out: pause the queue in output.
        :param kw_all: pause the queue in input and output (same as specifying
                       both the in and out options).
        :param kw_none: clear the paused state in input and output.
        :param kw_state: just report the current queue state.
        :param kw_bcast: send a PAUSE command to all the reachable nodes of
                         the cluster to set the same queue in the other nodes
                         to the same state.
        """
        command = ["PAUSE", queue_name]
        if kw_in:
            command += ["in"]
        if kw_out:
            command += ["out"]
        if kw_all:
            command += ["all"]
        if kw_none:
            command += ["none"]
        if kw_state:
            command += ["state"]
        if kw_bcast:
            command += ["bcast"]

        return self.execute_command(*command)
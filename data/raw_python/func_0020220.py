def wait(self):
        """
        If this isn't the master process, wait for instructions.

        """
        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        status = MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            if self.debug:
                print("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if self.debug:
                print("Worker {0} got task {1} with tag {2}."
                      .format(self.rank, type(task), status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                if self.debug:
                    print("Worker {0} told to quit.".format(self.rank))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, _function_wrapper):
                self.function = task.function
                if self.debug:
                    print("Worker {0} replaced its task function: {1}."
                          .format(self.rank, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            result = self.function(task)
            if self.debug:
                print("Worker {0} sending answer {1} with tag {2}."
                      .format(self.rank, type(result), status.tag))
            self.comm.isend(result, dest=0, tag=status.tag)

        # Kill the process?
        if self.exit_on_end:
            sys.exit()
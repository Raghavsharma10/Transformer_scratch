def map(self, function, tasks):
        """
        Like the built-in :py:func:`map` function, apply a function to all
        of the values in a list and return the list of results.

        :param function:
            The function to apply to the list.

        :param tasks:
            The list of elements.

        """
        ntask = len(tasks)

        # If not the master just wait for instructions.
        if not self.is_master():
            self.wait()
            return

        if function is not self.function:
            if self.debug:
                print("Master replacing pool function with {0}."
                      .format(function))

            self.function = function
            F = _function_wrapper(function)

            # Tell all the workers what function to use.
            requests = []
            for i in range(self.size):
                r = self.comm.isend(F, dest=i + 1)
                requests.append(r)

            # Wait until all of the workers have responded. See:
            #       https://gist.github.com/4176241
            MPI.Request.waitall(requests)

        if (not self.loadbalance) or (ntask <= self.size):
            # Do not perform load-balancing - the default load-balancing
            # scheme emcee uses.

            # Send all the tasks off and wait for them to be received.
            # Again, see the bug in the above gist.
            requests = []
            for i, task in enumerate(tasks):
                worker = i % self.size + 1
                if self.debug:
                    print("Sent task {0} to worker {1} with tag {2}."
                          .format(type(task), worker, i))
                r = self.comm.isend(task, dest=worker, tag=i)
                requests.append(r)

            MPI.Request.waitall(requests)

            # Now wait for the answers.
            results = []
            for i in range(ntask):
                worker = i % self.size + 1
                if self.debug:
                    print("Master waiting for worker {0} with tag {1}"
                          .format(worker, i))
                result = self.comm.recv(source=worker, tag=i)

                results.append(result)

            return results

        else:
            # Perform load-balancing. The order of the results are likely to
            # be different from the previous case.
            for i, task in enumerate(tasks[0:self.size]):
                worker = i + 1
                if self.debug:
                    print("Sent task {0} to worker {1} with tag {2}."
                          .format(type(task), worker, i))
                # Send out the tasks asynchronously.
                self.comm.isend(task, dest=worker, tag=i)

            ntasks_dispatched = self.size
            results = [None] * ntask
            for itask in range(ntask):
                status = MPI.Status()
                # Receive input from workers.
                try:
                    result = self.comm.recv(source=MPI.ANY_SOURCE,
                                            tag=MPI.ANY_TAG, status=status)
                except Exception as e:
                    self.close()
                    raise e

                worker = status.source
                i = status.tag
                results[i] = result
                if self.debug:
                    print("Master received from worker {0} with tag {1}"
                          .format(worker, i))

                # Now send the next task to this idle worker (if there are any
                # left).
                if ntasks_dispatched < ntask:
                    task = tasks[ntasks_dispatched]
                    i = ntasks_dispatched
                    if self.debug:
                        print("Sent task {0} to worker {1} with tag {2}."
                              .format(type(task), worker, i))
                    # Send out the tasks asynchronously.
                    self.comm.isend(task, dest=worker, tag=i)
                    ntasks_dispatched += 1

            return results
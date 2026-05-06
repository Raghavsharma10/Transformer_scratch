def doTask(self, task):
        """Filter input *task* to pipelines -- make sure each one has no more
        than *max_tasks* tasks in it. Return a tuple
          (*task*, *results*)
        where *task* is the given task, and *results* is 
        a list of latest retrieved results from pipelines."""

        # If we're not caching, then clear the table of last results.
        if not self._cache_results:
            self._last_results = dict()

        # Iterate the list of pipelines, draining each one of any results.
        # For pipelines whose current stream has less than *max_tasks* tasks 
        # remaining, feed them the current task.
        for pipe in self._pipelines:

            count = self._task_counts[pipe]

            # Let's attempt to drain all (if any) results from the pipeline.
            valid = True
            last_result = None
            while count and valid:
                valid, result = pipe.get(sys.float_info.min)
                if valid:
                    last_result = result
                    count -= 1

            # Unless we're dropping results, save the last result (if any).
            if not self._drop_results:
                if last_result is not None:
                    self._last_results[pipe] = last_result

            # If there is room for the task, or if it is a "stop" request,
            # put it on the pipeline.
            if count <= self._max_tasks-1 or task is None:
                pipe.put(task)
                count += 1

            # Update the task count for the pipeline.
            self._task_counts[pipe] = count

        # If we're only propagating the task, do so now.
        if self._drop_results:
            return task

        # Otherwise, also propagate the assembly of pipeline results.
        all_results = [res for res in self._last_results.values()]
        return task, all_results
def cross_entropy_error(self, input_data, targets, average=True,
                            cache=None, prediction=False,
                            sum_errors=True):
        """ Computes the cross-entropy error for all tasks.
        """

        loss = []
        if cache is None:
            cache = self.n_tasks * [None]

        for targets_task, cache_task, task in \
            izip(targets, cache, self.tasks):
            loss.append(task.cross_entropy_error(
                input_data, targets_task, average=average,
                cache=cache_task,
                prediction=prediction))

        if sum_errors:
            return sum(loss)
        else:
            return loss
def _run(self, data, store, signal, context, *,
             success_callback=None, stop_callback=None, abort_callback=None):
        """ The internal run method that decorates the public run method.

        This method makes sure data is being passed to and from the task.

        Args:
            data (MultiTaskData): The data object that has been passed from the
                                  predecessor task.
            store (DataStoreDocument): The persistent data store object that allows the
                                       task to store data for access across the current
                                       workflow run.
            signal (TaskSignal): The signal object for tasks. It wraps the construction
                                 and sending of signals into easy to use methods.
            context (TaskContext): The context in which the tasks runs.
            success_callback: This function is called when the task completed successfully
            stop_callback: This function is called when a StopTask exception was raised.
            abort_callback: This function is called when an AbortWorkflow exception
                            was raised.

        Raises:
            TaskReturnActionInvalid: If the return value of the task is not
                                     an Action object.

        Returns:
            Action: An Action object containing the data that should be passed on
                    to the next task and optionally a list of successor tasks that
                    should be executed.
        """
        if data is None:
            data = MultiTaskData()
            data.add_dataset(self._name)

        try:
            if self._callback_init is not None:
                self._callback_init(data, store, signal, context)

            result = self.run(data, store, signal, context)

            if self._callback_finally is not None:
                self._callback_finally(TaskStatus.Success, data, store, signal, context)

            if success_callback is not None:
                success_callback()

        # the task should be stopped and optionally all successor tasks skipped
        except StopTask as err:
            if self._callback_finally is not None:
                self._callback_finally(TaskStatus.Stopped, data, store, signal, context)

            if stop_callback is not None:
                stop_callback(exc=err)

            result = Action(data, limit=[]) if err.skip_successors else None

        # the workflow should be stopped immediately
        except AbortWorkflow as err:
            if self._callback_finally is not None:
                self._callback_finally(TaskStatus.Aborted, data, store, signal, context)

            if abort_callback is not None:
                abort_callback(exc=err)

            result = None
            signal.stop_workflow()

        # catch any other exception, call the finally callback, then re-raise
        except:
            if self._callback_finally is not None:
                self._callback_finally(TaskStatus.Error, data, store, signal, context)

            signal.stop_workflow()
            raise

        # handle the returned data (either implicitly or as an returned Action object) by
        # flattening all, possibly modified, input datasets in the MultiTask data down to
        # a single output dataset.
        if result is None:
            data.flatten(in_place=True)
            data.add_task_history(self.name)
            return Action(data)
        else:
            if not isinstance(result, Action):
                raise TaskReturnActionInvalid()

            result.data.flatten(in_place=True)
            result.data.add_task_history(self.name)
            return result
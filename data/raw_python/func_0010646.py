def run(self, config, workflow_id, signal, *, data=None):
        """ Run the dag by calling the tasks in the correct order.

        Args:
            config (Config): Reference to the configuration object from which the
                             settings for the dag are retrieved.
            workflow_id (str): The unique ID of the workflow that runs this dag.
            signal (DagSignal): The signal object for dags. It wraps the construction
                                and sending of signals into easy to use methods.
            data (MultiTaskData): The initial data that is passed on to the start tasks.

        Raises:
            DirectedAcyclicGraphInvalid: If the graph is not a dag (e.g. contains loops).
            ConfigNotDefinedError: If the configuration for the dag is empty.
        """
        graph = self.make_graph(self._schema)

        # pre-checks
        self.validate(graph)

        if config is None:
            raise ConfigNotDefinedError()

        # create the celery app for submitting tasks
        celery_app = create_app(config)

        # the task queue for managing the current state of the tasks
        tasks = []
        stopped = False

        # add all tasks without predecessors to the task list
        for task in nx.topological_sort(graph):
            task.workflow_name = self.workflow_name
            task.dag_name = self.name
            if len(list(graph.predecessors(task))) == 0:
                task.state = TaskState.Waiting
                tasks.append(task)

        def set_task_completed(completed_task):
            """ For each completed task, add all successor tasks to the task list.
            If they are not in the task list yet, flag them as 'waiting'.
            """
            completed_task.state = TaskState.Completed
            for successor in graph.successors(completed_task):
                if successor not in tasks:
                    successor.state = TaskState.Waiting
                    tasks.append(successor)

        # process the task queue as long as there are tasks in it
        while tasks:
            if not stopped:
                stopped = signal.is_stopped

            # delay the execution by the polling time
            if config.dag_polling_time > 0.0:
                sleep(config.dag_polling_time)

            for i in range(len(tasks) - 1, -1, -1):
                task = tasks[i]

                # for each waiting task, wait for all predecessor tasks to be
                # completed. Then check whether the task should be skipped by
                # interrogating the predecessor tasks.
                if task.is_waiting:
                    if stopped:
                        task.state = TaskState.Stopped
                    else:
                        pre_tasks = list(graph.predecessors(task))
                        if all([p.is_completed for p in pre_tasks]):

                            # check whether the task should be skipped
                            run_task = task.has_to_run or len(pre_tasks) == 0
                            for pre in pre_tasks:
                                if run_task:
                                    break

                                # predecessor task is skipped and flag should
                                # not be propagated
                                if pre.is_skipped and not pre.propagate_skip:
                                    run_task = True

                                # limits of a non-skipped predecessor task
                                if not pre.is_skipped:
                                    if pre.celery_result.result.limit is not None:
                                        if task.name in [
                                            n.name if isinstance(n, BaseTask) else n
                                                for n in pre.celery_result.result.limit]:
                                            run_task = True
                                    else:
                                        run_task = True

                            task.is_skipped = not run_task

                            # send the task to celery or, if skipped, mark it as completed
                            if task.is_skipped:
                                set_task_completed(task)
                            else:
                                # compose the input data from the predecessor tasks
                                # output. Data from skipped predecessor tasks do not
                                # contribute to the input data
                                if len(pre_tasks) == 0:
                                    input_data = data
                                else:
                                    input_data = MultiTaskData()
                                    for pt in [p for p in pre_tasks if not p.is_skipped]:
                                        slot = graph[pt][task]['slot']
                                        input_data.add_dataset(
                                            pt.name,
                                            pt.celery_result.result.data.default_dataset,
                                            aliases=[slot] if slot is not None else None)

                                task.state = TaskState.Running
                                task.celery_result = celery_app.send_task(
                                    JobExecPath.Task,
                                    args=(task, workflow_id, input_data),
                                    queue=task.queue,
                                    routing_key=task.queue
                                )

                # flag task as completed
                elif task.is_running:
                    if task.celery_completed:
                        set_task_completed(task)
                    elif task.celery_failed:
                        task.state = TaskState.Aborted
                        signal.stop_workflow()

                # cleanup task results that are not required anymore
                elif task.is_completed:
                    if all([s.is_completed or s.is_stopped or s.is_aborted
                            for s in graph.successors(task)]):
                        if celery_app.conf.result_expires == 0:
                            task.clear_celery_result()
                        tasks.remove(task)

                # cleanup and remove stopped and aborted tasks
                elif task.is_stopped or task.is_aborted:
                    if celery_app.conf.result_expires == 0:
                        task.clear_celery_result()
                    tasks.remove(task)
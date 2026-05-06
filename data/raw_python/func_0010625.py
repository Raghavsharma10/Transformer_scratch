def execute_task(self, task, workflow_id, data=None):
    """ Celery task that runs a single task on a worker.

    Args:
        self (Task): Reference to itself, the celery task object.
        task (BaseTask): Reference to the task object that performs the work
                         in its run() method.
        workflow_id (string): The unique ID of the workflow run that started this task.
        data (MultiTaskData): An optional MultiTaskData object that contains the data
                              that has been passed down from upstream tasks.
    """
    start_time = datetime.utcnow()

    store_doc = DataStore(**self.app.user_options['config'].data_store,
                          auto_connect=True).get(workflow_id)
    store_loc = 'log.{}.tasks.{}'.format(task.dag_name, task.name)

    def handle_callback(message, event_type, exc=None):
        msg = '{}: {}'.format(message, str(exc)) if exc is not None else message

        # set the logging level
        if event_type == JobEventName.Stopped:
            logger.warning(msg)
        elif event_type == JobEventName.Aborted:
            logger.error(msg)
        else:
            logger.info(msg)

        current_time = datetime.utcnow()

        # store provenance information about a task
        if event_type != JobEventName.Started:
            duration = (current_time - start_time).total_seconds()

            store_doc.set(key='{}.end_time'.format(store_loc),
                          value=current_time,
                          section=DataStoreDocumentSection.Meta)

            store_doc.set(key='{}.duration'.format(store_loc),
                          value=duration,
                          section=DataStoreDocumentSection.Meta)
        else:
            # store provenance information about a task
            store_doc.set(key='{}.start_time'.format(store_loc),
                          value=start_time,
                          section=DataStoreDocumentSection.Meta)

            store_doc.set(key='{}.worker'.format(store_loc),
                          value=self.request.hostname,
                          section=DataStoreDocumentSection.Meta)

            store_doc.set(key='{}.queue'.format(store_loc),
                          value=task.queue,
                          section=DataStoreDocumentSection.Meta)
            duration = None

        # send custom celery event
        self.send_event(event_type,
                        job_type=JobType.Task,
                        name=task.name,
                        queue=task.queue,
                        time=current_time,
                        workflow_id=workflow_id,
                        duration=duration)

    # store job specific meta information wth the job
    self.update_state(meta={'name': task.name,
                            'queue': task.queue,
                            'type': JobType.Task,
                            'workflow_id': workflow_id})

    # send start celery event
    handle_callback('Start task <{}>'.format(task.name), JobEventName.Started)

    # run the task and capture the result
    return task._run(
        data=data,
        store=store_doc,
        signal=TaskSignal(Client(
            SignalConnection(**self.app.user_options['config'].signal, auto_connect=True),
            request_key=workflow_id),
            task.dag_name),
        context=TaskContext(task.name, task.dag_name, task.workflow_name,
                            workflow_id, self.request.hostname),
        success_callback=partial(handle_callback,
                                 message='Complete task <{}>'.format(task.name),
                                 event_type=JobEventName.Succeeded),
        stop_callback=partial(handle_callback,
                              message='Stop task <{}>'.format(task.name),
                              event_type=JobEventName.Stopped),
        abort_callback=partial(handle_callback,
                               message='Abort workflow <{}> by task <{}>'.format(
                                   task.workflow_name, task.name),
                               event_type=JobEventName.Aborted))
def execute_workflow(self, workflow, workflow_id=None):
    """ Celery task (aka job) that runs a workflow on a worker.

    This celery task starts, manages and monitors the dags that make up a workflow.

    Args:
        self (Task): Reference to itself, the celery task object.
        workflow (Workflow): Reference to the workflow object that is being used to
                             start, manage and monitor dags.
        workflow_id (string): If a workflow ID is provided the workflow run will use
                              this ID, if not a new ID will be auto generated.
    """
    start_time = datetime.utcnow()

    logger.info('Running workflow <{}>'.format(workflow.name))
    data_store = DataStore(**self.app.user_options['config'].data_store,
                           auto_connect=True)

    # create a unique workflow id for this run
    if data_store.exists(workflow_id):
        logger.info('Using existing workflow ID: {}'.format(workflow_id))
    else:
        workflow_id = data_store.add(payload={
                                         'name': workflow.name,
                                         'queue': workflow.queue,
                                         'start_time': start_time
                                     })
        logger.info('Created workflow ID: {}'.format(workflow_id))

    # send custom celery event that the workflow has been started
    self.send_event(JobEventName.Started,
                    job_type=JobType.Workflow,
                    name=workflow.name,
                    queue=workflow.queue,
                    time=start_time,
                    workflow_id=workflow_id,
                    duration=None)

    # create server for inter-task messaging
    signal_server = Server(SignalConnection(**self.app.user_options['config'].signal,
                                            auto_connect=True),
                           request_key=workflow_id)

    # store job specific meta information wth the job
    self.update_state(meta={'name': workflow.name,
                            'type': JobType.Workflow,
                            'workflow_id': workflow_id,
                            'queue': workflow.queue,
                            'start_time': start_time,
                            'arguments': workflow.provided_arguments})

    # run the DAGs in the workflow
    workflow.run(config=self.app.user_options['config'],
                 data_store=data_store,
                 signal_server=signal_server,
                 workflow_id=workflow_id)

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    # update data store with provenance information
    store_doc = data_store.get(workflow_id)
    store_doc.set(key='end_time', value=end_time,
                  section=DataStoreDocumentSection.Meta)
    store_doc.set(key='duration', value=duration,
                  section=DataStoreDocumentSection.Meta)

    # send custom celery event that the workflow has succeeded
    event_name = JobEventName.Succeeded if not workflow.is_stopped \
        else JobEventName.Aborted

    self.send_event(event_name,
                    job_type=JobType.Workflow,
                    name=workflow.name,
                    queue=workflow.queue,
                    time=end_time,
                    workflow_id=workflow_id,
                    duration=duration)

    logger.info('Finished workflow <{}>'.format(workflow.name))
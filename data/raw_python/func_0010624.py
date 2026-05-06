def execute_dag(self, dag, workflow_id, data=None):
    """ Celery task that runs a single dag on a worker.

    This celery task starts, manages and monitors the individual tasks of a dag.

    Args:
        self (Task): Reference to itself, the celery task object.
        dag (Dag): Reference to a Dag object that is being used to start, manage and
                   monitor tasks.
        workflow_id (string): The unique ID of the workflow run that started this dag.
        data (MultiTaskData): An optional MultiTaskData object that is being passed to
                              the first tasks in the dag. This allows the transfer of
                              data from dag to dag.
    """
    start_time = datetime.utcnow()
    logger.info('Running DAG <{}>'.format(dag.name))

    store_doc = DataStore(**self.app.user_options['config'].data_store,
                          auto_connect=True).get(workflow_id)
    store_loc = 'log.{}'.format(dag.name)

    # update data store with provenance information
    store_doc.set(key='{}.start_time'.format(store_loc), value=start_time,
                  section=DataStoreDocumentSection.Meta)

    # send custom celery event that the dag has been started
    self.send_event(JobEventName.Started,
                    job_type=JobType.Dag,
                    name=dag.name,
                    queue=dag.queue,
                    time=start_time,
                    workflow_id=workflow_id,
                    duration=None)

    # store job specific meta information wth the job
    self.update_state(meta={'name': dag.name,
                            'queue': dag.queue,
                            'type': JobType.Dag,
                            'workflow_id': workflow_id})

    # run the tasks in the DAG
    signal = DagSignal(Client(SignalConnection(**self.app.user_options['config'].signal,
                                               auto_connect=True),
                              request_key=workflow_id), dag.name)
    dag.run(config=self.app.user_options['config'],
            workflow_id=workflow_id,
            signal=signal,
            data=data)

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    # update data store with provenance information
    store_doc.set(key='{}.end_time'.format(store_loc), value=end_time,
                  section=DataStoreDocumentSection.Meta)
    store_doc.set(key='{}.duration'.format(store_loc), value=duration,
                  section=DataStoreDocumentSection.Meta)

    # send custom celery event that the dag has succeeded
    event_name = JobEventName.Succeeded if not signal.is_stopped else JobEventName.Aborted
    self.send_event(event_name,
                    job_type=JobType.Dag,
                    name=dag.name,
                    queue=dag.queue,
                    time=end_time,
                    workflow_id=workflow_id,
                    duration=duration)

    logger.info('Finished DAG <{}>'.format(dag.name))
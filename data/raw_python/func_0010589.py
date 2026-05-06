def run(self, config, data_store, signal_server, workflow_id):
        """ Run all autostart dags in the workflow.

        Only the dags that are flagged as autostart are started.

        Args:
            config (Config): Reference to the configuration object from which the
                             settings for the workflow are retrieved.
            data_store (DataStore): A DataStore object that is fully initialised and
                        connected to the persistent data storage.
            signal_server (Server): A signal Server object that receives requests
                                    from dags and tasks.
            workflow_id (str): A unique workflow id that represents this workflow run
        """
        self._workflow_id = workflow_id
        self._celery_app = create_app(config)

        # pre-fill the data store with supplied arguments
        args = self._parameters.consolidate(self._provided_arguments)
        for key, value in args.items():
            data_store.get(self._workflow_id).set(key, value)

        # start all dags with the autostart flag set to True
        for name, dag in self._dags_blueprint.items():
            if dag.autostart:
                self._queue_dag(name)

        # as long as there are dags in the list keep running
        while self._dags_running:
            if config.workflow_polling_time > 0.0:
                sleep(config.workflow_polling_time)

            # handle new requests from dags, tasks and the library (e.g. cli, web)
            for i in range(MAX_SIGNAL_REQUESTS):
                request = signal_server.receive()
                if request is None:
                    break

                try:
                    response = self._handle_request(request)
                    if response is not None:
                        signal_server.send(response)
                    else:
                        signal_server.restore(request)
                except (RequestActionUnknown, RequestFailed):
                    signal_server.send(Response(success=False, uid=request.uid))

            # remove any dags and their result data that finished running
            for name, dag in list(self._dags_running.items()):
                if dag.ready():
                    if self._celery_app.conf.result_expires == 0:
                        dag.forget()
                    del self._dags_running[name]
                elif dag.failed():
                    self._stop_workflow = True

        # remove the signal entry
        signal_server.clear()

        # delete all entries in the data_store under this workflow id, if requested
        if self._clear_data_store:
            data_store.remove(self._workflow_id)
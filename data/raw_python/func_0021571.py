def spawn_worker(params):
    """
    This method has to be module level function

    :type params: Params
    """
    setup_logging(params)
    log.info("Adding worker: idx=%s\tconcurrency=%s\tresults=%s", params.worker_index, params.concurrency,
             params.report)
    worker = Worker(params)
    worker.start()
    worker.join()
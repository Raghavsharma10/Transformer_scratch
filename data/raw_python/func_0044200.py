def make_server(function, port, authkey, qsize=None):
    """Create a manager containing input and output queues, and a function
    to map inputs over. A connecting client can read the stored function,
    apply it to items in the input queue and post back to the output
    queue

    :param function: function to apply to inputs
    :param port: port over which to server
    :param authkey: authorization key

    """
    QueueManager.register('get_job_q',
        callable=partial(return_arg, Queue(maxsize=qsize)))
    QueueManager.register('get_result_q',
        callable=partial(return_arg, Queue(maxsize=qsize)))
    QueueManager.register('get_function',
        callable=partial(return_arg, function))
    QueueManager.register('q_closed',
        callable=partial(return_arg, SharedConst(False)))

    # on windows host='' doesn't work, but 'localhost' breaks
    #   remote connections. Documentation terrible in this respect.
    #   So we're not supporting distributed compute on windows.
    host = 'localhost' if os.name == 'nt' else ''
    manager = QueueManager(address=(host, port), authkey=authkey)
    manager.start()
    return manager
def make_client(ip, port, authkey):
    """Create a manager to connect to our server manager

    :param ip: ip address of server
    :param port: port over which to server
    :param authkey: authorization key

    """
    QueueManager.register('get_job_q')
    QueueManager.register('get_result_q')
    QueueManager.register('get_function')
    QueueManager.register('q_closed')

    manager = QueueManager(address=(ip, port), authkey=authkey)
    manager.connect()
    return manager
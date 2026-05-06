def run_client(ip, port, authkey, max_items=None, timeout=2):
    """Connect to a SwarmServer and do its dirty work.

    :param ip: ip address of server
    :param port: port to connect to on server
    :param authkey: authorization key
    :param max_items: maximum number of items to process from server.
        Useful for say running clients on a cluster.
    """

    manager = make_client(ip, port, authkey)
    job_q = manager.get_job_q()
    job_q_closed = manager.q_closed()
    result_q = manager.get_result_q()
    function = manager.get_function()._getvalue()

    processed = 0
    while True:
        try:
            job = job_q.get_nowait()
            result = function(job)
            result_q.put(result)
        except Queue.Empty:
            if job_q_closed._getvalue().value:
                break
        else:
            processed += 1
            if max_items is not None and processed == max_items:
                break
        sleep(timeout)
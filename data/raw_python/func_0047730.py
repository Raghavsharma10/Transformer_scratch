def monitor(name, callback):
    '''
    monitors actions on the specified container,
    callback is a function to be called on 
    '''
    global _monitor 
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    if _monitor:
        if _monitor.is_monitored(name):
            raise Exception("You are already monitoring this container (%s)" % name)
    else:
        _monitor = _LXCMonitor()
        logging.info("Starting LXC Monitor")
        _monitor.start()
        def kill_handler(sg, fr):
            stop_monitor()
        signal.signal(signal.SIGTERM, kill_handler)
        signal.signal(signal.SIGINT, kill_handler)
    _monitor.add_monitor(name, callback)
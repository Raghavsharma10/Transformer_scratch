def notify(name, states, callback):
    '''
    executes the callback function with no parameters when the container reaches the specified state or states
    states can be or-ed or and-ed
        notify('test', 'STOPPED', letmeknow)
        
        notify('test', 'STOPPED|RUNNING', letmeknow)
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)

    cmd = ['lxc-wait', '-n', name, '-s', states]
    def th():
        subprocess.check_call(cmd)
        callback()
    _logger.info("Waiting on states %s for container %s", states, name)
    threading.Thread(target=th).start()
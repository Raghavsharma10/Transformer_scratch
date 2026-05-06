def shutdown(name, wait=False, reboot=False):
    '''
    graceful shutdown sent to the container
    :param wait: should we wait for the shutdown to complete?
    :param reboot: reboot a container, ignores wait
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-shutdown', '-n', name]
    if wait:
        cmd += ['-w']
    if reboot:
        cmd += ['-r']
        
    subprocess.check_call(cmd)
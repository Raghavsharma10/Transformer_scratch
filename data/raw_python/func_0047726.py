def stop(name):
    '''
    stops a container
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-stop', '-n', name]
    subprocess.check_call(cmd)
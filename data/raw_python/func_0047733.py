def info(name):
    '''
    returns info dict about the specified container
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-info', '-n', name]
    out = subprocess.check_output(cmd).splitlines()
    info = {}
    for line in out:
        k, v = line.split()
        info[k] = v
    return info
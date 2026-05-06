def kill(name, signal):
    '''
    sends a kill signal to process 1 of ths container <name>
    :param signal: numeric signal
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-kill', '--name=%s' % name, signal]
    subprocess.check_call(cmd)
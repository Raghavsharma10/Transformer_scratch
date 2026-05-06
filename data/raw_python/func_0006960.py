def wait_for_ssh(host, port=22, timeout=600):
    """ probes the ssh port and waits until it is available """
    log_yellow('waiting for ssh...')
    for iteration in xrange(1, timeout): #noqa
        sleep(1)
        if is_ssh_available(host, port):
            return True
        else:
            log_yellow('waiting for ssh...')
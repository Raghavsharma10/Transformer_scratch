def all_as_dict():
    '''
    returns a dict {'Running': ['cont1', 'cont2'], 
                    'Stopped': ['cont3', 'cont4']
                    }
                    
    '''
    cmd = ['lxc-list']
    out = subprocess.check_output(cmd).splitlines()
    stopped = []
    running = []
    frozen = []
    current = None
    for c in out:
        c = c.strip()
        if c == 'RUNNING':
            current = running
            continue
        if c == 'STOPPED':
            current = stopped
            continue
        if c == 'FROZEN':
            current = frozen
            continue
        if not len(c):
            continue
        current.append(c)
    return {'Running': running,
            'Stopped': stopped,
            'Frozen': frozen}
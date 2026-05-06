def driver_send(command,hostname=None,wait=0.2):
    '''Send a command (or ``list`` of commands) to AFNI at ``hostname`` (defaults to local host)
    Requires plugouts enabled (open afni with ``-yesplugouts`` or set ``AFNI_YESPLUGOUTS = YES`` in ``.afnirc``)
    If ``wait`` is not ``None``, will automatically sleep ``wait`` seconds after sending the command (to make sure it took effect)'''
    cmd = ['plugout_drive']
    if hostname:
        cmd += ['-host',hostname]
    if isinstance(command,basestring):
        command = [command]
    cmd += [['-com',x] for x in command] + ['-quit']
    o = nl.run(cmd,quiet=None,stderr=None)
    if wait!=None:
        time.sleep(wait)
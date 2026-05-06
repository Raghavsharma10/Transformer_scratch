def pipe_shell_cmds(shell_CMDs):
    '''
        shell_CMDs = {}
        shell_CMDs[1] = 'netstat -n'
        shell_CMDs[2] = "awk {'print $6'}"
    '''
    len = shell_CMDs.__len__()
    p = {}
    p[1] = subprocess.Popen(shlex.split(shell_CMDs[1]), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(2,len):
        p[i] = subprocess.Popen(shlex.split(shell_CMDs[i]), stdin=p[i-1].stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if(len > 1):
        p[len] = subprocess.Popen(shlex.split(shell_CMDs[len]), stdin=p[len-1].stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    result = p[len].communicate()
    if(len > 1):
        for i in range(2,len+1):
            returncode = p[i].wait()
    else:
        returncode = p[len].wait()
    return(result)
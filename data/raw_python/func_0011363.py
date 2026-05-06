def run_command(cmd):
    '''run_command uses subprocess to send a command to the terminal.
    :param cmd: the command to send, should be a list for subprocess
    '''
    output = Popen(cmd,stderr=STDOUT,stdout=PIPE)
    t = output.communicate()[0],output.returncode
    output = {'message':t[0],
              'return_code':t[1]}

    return output
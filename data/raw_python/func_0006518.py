def cmd2list(cmd):
   ''' Executes a command through the operating system and returns the output
   as a list, or on error a string with the standard error.
   EXAMPLE:
      >>> from subprocess import Popen, PIPE
      >>> CMDout2array('ls -l')
   '''
   p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
   stdout, stderr = p.communicate()
   if p.returncode != 0 and stderr != '':
      return "ERROR: %s\n"%(stderr)
   else:
      return stdout.split('\n')
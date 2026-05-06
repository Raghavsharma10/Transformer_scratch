def execute_no_wait(self, cmd, walltime, envs={}):
        ''' Synchronously execute a commandline string on the shell.

        Args:
            - cmd (string) : Commandline string to execute
            - walltime (int) : walltime in seconds, this is not really used now.

        Returns:

           - retcode : Return code from the execution, -1 on fail
           - stdout  : stdout string
           - stderr  : stderr string

        Raises:
         None.
        '''
        current_env = copy.deepcopy(self._envs)
        current_env.update(envs)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.userhome,
                env=current_env,
                shell=True,
                preexec_fn=os.setpgrp
            )
            pid = proc.pid

        except Exception as e:
            print("Caught exception : {0}".format(e))
            logger.warn("Execution of command [%s] failed due to \n %s ", (cmd, e))

        return pid, proc
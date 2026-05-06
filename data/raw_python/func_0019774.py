def execProcCmd(self, *args):
        """Execute ps command with positional params args and return result as 
        list of lines.
        
        @param *args: Positional params for ps command.
        @return:      List of output lines
        
        """
        out = util.exec_command([psCmd,] + list(args))
        return out.splitlines()
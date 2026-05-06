def execNetstatCmd(self, *args):
        """Execute ps command with positional params args and return result as 
        list of lines.
        
        @param *args: Positional params for netstat command.
        @return:      List of output lines
        
        """
        out = util.exec_command([netstatCmd,] + list(args))
        return out.splitlines()
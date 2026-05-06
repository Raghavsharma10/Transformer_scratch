def _execCmd(self, cmd, args):
        """Execute command and return result body as list of lines.
        
            @param cmd:  Command string.
            @param args: Comand arguments string. 
            @return:     Result dictionary.
            
        """
        output = self._eslconn.api(cmd, args)
        if output:
            body = output.getBody()
            if body:
                return body.splitlines()
        return None
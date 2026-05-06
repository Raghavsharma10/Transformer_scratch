def _execShowCountCmd(self, showcmd):
        """Execute 'show' command and return result dictionary.
        
            @param cmd: Command string.
            @return: Result dictionary.
            
        """
        result = None
        lines = self._execCmd("show", showcmd + " count")
        for line in lines:
            mobj = re.match('\s*(\d+)\s+total', line)
            if mobj:
                return int(mobj.group(1))
        return result
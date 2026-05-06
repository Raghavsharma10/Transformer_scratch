def _execShowCmd(self, showcmd):
        """Execute 'show' command and return result dictionary.
        
            @param cmd: Command string.
            @return: Result dictionary.
            
        """
        result = None
        lines = self._execCmd("show", showcmd)
        if lines and len(lines) >= 2 and lines[0] != '' and lines[0][0] != '-':
            result = {}
            result['keys'] = lines[0].split(',')
            items = []
            for line in lines[1:]:
                if line == '':
                    break
                items.append(line.split(','))
            result['items'] = items
        return result
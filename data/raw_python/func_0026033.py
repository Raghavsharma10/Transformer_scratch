def save(self, dolist=0):
        """Return .par format string for this parameter

        If dolist is set, returns fields as a list of strings.  Default
        is to return a single string appropriate for writing to a file.
        """
        quoted = not dolist
        fields = 7*[""]
        fields[0] = self.name
        fields[1] = self.type
        fields[2] = self.mode
        fields[3] = self.toString(self.value,quoted=quoted)
        if self.choice is not None:
            schoice = list(map(self.toString, self.choice))
            schoice.insert(0,'')
            schoice.append('')
            fields[4] = repr('|'.join(schoice))
        elif self.min not in [None,INDEF]:
            fields[4] = self.toString(self.min,quoted=quoted)
        if self.max not in [None,INDEF]:
            fields[5] = self.toString(self.max,quoted=quoted)
        if self.prompt:
            if quoted:
                sprompt = repr(self.prompt)
            else:
                sprompt = self.prompt
            # prompt can have embedded newlines (which are printed)
            sprompt = sprompt.replace(r'\012', '\n')
            sprompt = sprompt.replace(r'\n', '\n')
            fields[6] = sprompt
        # delete trailing null parameters
        for i in [6,5,4]:
            if fields[i] != "": break
            del fields[i]
        if dolist:
            return fields
        else:
            return ','.join(fields)
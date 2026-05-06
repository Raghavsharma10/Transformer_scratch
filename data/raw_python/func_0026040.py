def save(self, dolist=0):
        """Return .par format string for this parameter

        If dolist is set, returns fields as a list of strings.  Default
        is to return a single string appropriate for writing to a file.
        """
        quoted = not dolist
        array_size = 1
        for d in self.shape:
            array_size = d*array_size
        ndim = len(self.shape)
        fields = (7+2*ndim+len(self.value))*[""]
        fields[0] = self.name
        fields[1] = self.type
        fields[2] = self.mode
        fields[3] = str(ndim)
        next = 4
        for d in self.shape:
            fields[next] = str(d); next += 1
            fields[next] = '1';    next += 1
        nvstart = 7+2*ndim
        if self.choice is not None:
            schoice = list(map(self.toString, self.choice))
            schoice.insert(0,'')
            schoice.append('')
            fields[nvstart-3] = repr('|'.join(schoice))
        elif self.min not in [None,INDEF]:
            fields[nvstart-3] = self.toString(self.min,quoted=quoted)
        # insert an escaped line break before min field
        if quoted:
            fields[nvstart-3] = '\\\n' + fields[nvstart-3]
        if self.max not in [None,INDEF]:
            fields[nvstart-2] = self.toString(self.max,quoted=quoted)
        if self.prompt:
            if quoted:
                sprompt = repr(self.prompt)
            else:
                sprompt = self.prompt
            # prompt can have embedded newlines (which are printed)
            sprompt = sprompt.replace(r'\012', '\n')
            sprompt = sprompt.replace(r'\n', '\n')
            fields[nvstart-1] = sprompt
        for i in range(len(self.value)):
            fields[nvstart+i] = self.toString(self.value[i],quoted=quoted)
        # insert an escaped line break before value fields
        if dolist:
            return fields
        else:
            fields[nvstart] = '\\\n' + fields[nvstart]
            return ','.join(fields)
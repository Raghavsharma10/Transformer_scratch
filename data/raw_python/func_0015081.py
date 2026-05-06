def _parse_sensorupdate(self, msg):
        """
        Given a sensor-update message, returns the sensors/variables that were
        updated as a dict that maps sensors/variables to their updated values.
        """
        update = msg[self.sensorupdate_prefix_len:]
        parsed = [] # each element is either a sensor (key) or a sensor value
        curr_seg = '' # current segment (i.e. key or value) being formed
        numq = 0 # number of double quotes in current segment
        for seg in update.split(' ')[:-1]: # last char in update is a space
            numq += seg.count('"')
            curr_seg += seg
            # even number of quotes means we've finished parsing a segment
            if numq % 2 == 0: 
                parsed.append(curr_seg)
                curr_seg = ''
                numq = 0
            else: # segment has a space inside, so add back it in
                curr_seg += ' '
        unescaped = [self._unescape(self._get_type(x)) for x in parsed]
        # combine into a dict using iterators (both elements in the list
        # inputted to izip have a reference to the same iterator). even 
        # elements are keys, odd are values
        return dict(itertools.izip(*[iter(unescaped)]*2))
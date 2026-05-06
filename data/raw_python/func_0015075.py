def _pack(self, msg):
        """
        Packages msg according to Scratch message specification (encodes and 
        appends length prefix to msg). Credit to chalkmarrow from the 
        scratch.mit.edu forums for the prefix encoding code.
        """
        n = len(msg) 
        a = array.array('c')
        a.append(chr((n >> 24) & 0xFF))
        a.append(chr((n >> 16) & 0xFF))
        a.append(chr((n >>  8) & 0xFF))
        a.append(chr(n & 0xFF))
        return a.tostring() + msg
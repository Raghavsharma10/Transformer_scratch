def addr(self,address):
        '''returns dictionary with frame information for given address (a tuple of two hex numbers)'''
        if isinstance(address,basestring):
            # If you just gave me a single string, assume its "XXXX XXXX"
            addr = address.split()
        else:
            addr = list(address)
        # Convert to actual hex if you give me strings
        for i in xrange(len(addr)):
            if isinstance(addr[i],basestring):
                addr[i] = int(addr[i],16)
        for frame in self.raw_frames:
            if frame['addr']==address:
                return frame
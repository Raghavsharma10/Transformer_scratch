def getStringPartition(self):
        """
        Get the string representation of the current partition
        @return string like ":-1,0:2"
        """
        res = ''
        for s in self.partitions[self.index].getSlice():
            start = ''
            stop = ''
            if s.start is not None:
                start = int(s.start)
            if s.stop is not None:
                stop = int(s.stop)
            res += '{0}:{1},'.format(start, stop)
        return res
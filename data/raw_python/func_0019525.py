def total_count(self, seqs, nreport=100, scan_rc=True):
        """
        count the number of matches above the cutoff
        returns an iterator of lists containing integer counts
        """
        
        count_table = [counts for counts in self.count(seqs, nreport, scan_rc)]
        return np.sum(np.array(count_table), 0)
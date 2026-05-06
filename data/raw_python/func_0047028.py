def query_sequence_length(self):
    """ does not include hard clipped"""
    if self.entries.seq: return len(self.entries.seq)
    if not self.entries.cigar:
       raise ValueError('Cannot give a query length if no cigar and no query sequence are present')
    return sum([x[0] for x in self.cigar_array if re.match('[MIS=X]',x[1])])
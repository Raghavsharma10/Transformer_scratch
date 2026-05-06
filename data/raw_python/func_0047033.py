def cigar_array(self):
     """cache this one to speed things up a bit"""
     if self._cigar: return self._cigar
     self._cigar = [CIGARDatum(int(m[0]),m[1]) for m in re.findall('([0-9]+)([MIDNSHP=X]+)',self.entries.cigar)]
     return self._cigar
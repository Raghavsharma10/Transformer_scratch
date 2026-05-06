def alignment_ranges(self):
     """Put the heavy alignment ranges calculation called on demand and then cached"""
     if not self.is_aligned(): raise ValueError("you can't get alignment ranges from something that didn't align")
     if self._alignment_ranges: return self._alignment_ranges
     self._alignment_ranges = self._get_alignment_ranges()
     return self._alignment_ranges
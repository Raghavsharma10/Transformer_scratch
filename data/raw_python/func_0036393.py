def zero_crossing_before(self, n):
        """Find nearest zero crossing in waveform before frame ``n``"""
        n_in_samples = int(n * self.samplerate)

        search_start = n_in_samples - self.samplerate
        if search_start < 0:
            search_start = 0

        frame = zero_crossing_last(
            self.range_as_mono(search_start, n_in_samples)) + search_start

        return frame / float(self.samplerate)
def get_samples(self, sample_count):
        """
        Fetch a number of samples from self.wave_cache

        Args:
            sample_count (int): Number of samples to fetch

        Returns: ndarray
        """
        if self.amplitude.value <= 0:
            return None
        # Build samples by rolling the period cache through the buffer
        rolled_array = numpy.roll(self.wave_cache,
                                  -1 * self.last_played_sample)
        # Append remaining partial period
        full_count, remainder = divmod(sample_count, self.cache_length)
        final_subarray = rolled_array[:int(remainder)]
        return_array = numpy.concatenate((numpy.tile(rolled_array, full_count),
                                          final_subarray))
        # Keep track of where we left off to prevent popping between chunks
        self.last_played_sample = int(((self.last_played_sample + remainder) %
                                       self.cache_length))
        # Multiply output by amplitude
        return return_array * (self.amplitude.value *
                               self.amplitude_multiplier)
def get_counts_array_index(self, value):
        '''Return the index in the counts array for a given value
        '''
        if value < 0:
            raise ValueError("Histogram recorded value cannot be negative.")

        bucket_index = self._get_bucket_index(value)
        sub_bucket_index = self._get_sub_bucket_index(value, bucket_index)
        # Calculate the index for the first entry in the bucket:
        bucket_base_index = (bucket_index + 1) << self.sub_bucket_half_count_magnitude
        # The following is the equivalent of ((bucket_index + 1) * sub_bucket_half_count)
        # Calculate the offset in the bucket (can be negative for first bucket):
        offset_in_bucket = sub_bucket_index - self.sub_bucket_half_count
        # The following is the equivalent of
        # ((sub_bucket_index  - sub_bucket_half_count) + bucket_base_index
        return bucket_base_index + offset_in_bucket
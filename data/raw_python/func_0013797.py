def _is_common_binary(self, inpath):
        """private method to compare file path mime type to common binary file types"""
        # make local variables for the available char numbers in the suffix types to be tested
        two_suffix = inpath[-3:]
        three_suffix = inpath[-4:]
        four_suffix = inpath[-5:]
        
        # test for inclusion in the instance variable common_binaries (defined in __init__)
        if two_suffix in self.common_binaries:
            return True
        elif three_suffix in self.common_binaries:
            return True
        elif four_suffix in self.common_binaries:
            return True
        else:
            return False
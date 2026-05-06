def _is_common_text(self, inpath):
        """private method to compare file path mime type to common text file types"""
        # make local variables for the available char numbers in the suffix types to be tested
        one_suffix = inpath[-2:]
        two_suffix = inpath[-3:]
        three_suffix = inpath[-4:]
        four_suffix = inpath[-5:]
        
        # test for inclusion in the instance variable common_text (defined in __init__)
        if one_suffix in self.common_text:
            return True
        elif two_suffix in self.common_text:
            return True
        elif three_suffix in self.common_text:
            return True
        elif four_suffix in self.common_text:
            return True
        else:
            return False
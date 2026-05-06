def clean_super_features(self):
        """
        Removes any null & non-integer values from the super feature list
        """
        if self.super_features:
            self.super_features = [int(sf) for sf in self.super_features
                                   if sf is not None and is_valid_digit(sf)]
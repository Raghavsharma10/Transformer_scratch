def feature_handler(self, cmd):
        """Process a FeatureCommand."""
        feature = cmd.feature_name
        if feature not in commands.FEATURE_NAMES:
            self.warning("feature %s is not supported - parsing may fail"
                % (feature,))
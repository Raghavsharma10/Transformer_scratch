def check_order(self):
        """
        Performs the check and store the violations in self.violations.
        :return: boolean indicating the error state
        """

        for feature, info in self.constraints.items():
            self._check_feature(feature, info, 'before')
            self._check_feature(feature, info, 'after')
            self._check_position(feature, info)

        return not self.has_errors()
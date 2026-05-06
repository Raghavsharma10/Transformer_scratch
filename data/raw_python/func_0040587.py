def _check_feature(self, feature, info, mode):
        """
        Private helper method performing the order check.
        :param feature: the feature to check.
        :param info: the info dict containing the before and after constraints
        :param mode: after | before string
        :return: None
        """

        op = dict(
            before=operator.gt,
            after=operator.lt
        )[mode]

        feature_pos = self.get_feature_position(feature)

        if feature_pos is not None:
            # only proceed if the the feature exists in the current feature list

            for other in info.get(mode, []):
                other_pos = self.get_feature_position(other)

                if other_pos is not None:
                    # only proceed if the the other feature exists in the current feature list
                    if op(feature_pos, other_pos):
                        message = '{feature} (pos {feature_pos}) must be {mode} feature {other} (pos {other_pos}) but isn\'t.'.format(
                            feature=feature,
                            feature_pos=feature_pos,
                            other=other,
                            other_pos=other_pos,
                            mode=mode.upper()
                        )
                        self.violations.append((feature, message))
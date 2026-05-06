def _check_position(self, feature, info):
        """
        Takes the featur and the info dict and checks for the forced position
        :param feature:
        :param info:
        :return:
        """
        pos = info.get('position')
        if pos is not None:
            feature_pos = self.get_feature_position(feature)
            if feature_pos is not None:
                if feature_pos != pos:
                    message = '{feature} has a forced position on ({pos}) but is on position {feature_pos}.'.format(
                        feature=feature,
                        pos=pos,
                        feature_pos=feature_pos
                    )
                    self.violations.append((feature, message))
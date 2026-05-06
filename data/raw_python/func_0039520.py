def is_valid(self):
        """
        Checks the feature list product spec against.
        Checks if all mandartory features are contained;
        Checks that all "never" features are not contained
        :return: boolean
        """
        for spec in self.product_specs:

            for feature in spec.get('mandatory', []):
                if feature.replace('__', '.') not in self.feature_list:
                    self.errors_mandatory.append(feature)

            for feature in spec.get('never', []):
                if feature.replace('__', '.') in self.feature_list:
                    self.errors_never.append(feature)

        return not self.has_errors()
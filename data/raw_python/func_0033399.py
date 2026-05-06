def export_features(self, delimiter='\\'):
        """Export the features from this namespace as a list of feature tokens
        with the namespace name prepended (delimited by 'delimiter').

        Example:
        >>> namespace = Namespace('name1', features=['feature1', 'feature2'])
        >>> print namespace.export_features()
        ['name1\\feature1', 'name1\\feature2']
        """
        result_list = []
        for feature in self.features:
            result = '{}{}{}'.format(self.name, delimiter, feature)
            result_list.append(result)
        return result_list
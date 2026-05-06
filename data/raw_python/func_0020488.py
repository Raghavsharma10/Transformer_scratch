def get_build_config_by_labels(self, label_selectors):
        """
        Returns a build config matching the given label
        selectors. This method will raise OsbsException
        if not exactly one build config is found.
        """
        items = self.get_all_build_configs_by_labels(label_selectors)

        if not items:
            raise OsbsException(
                "Build config not found for labels: %r" %
                (label_selectors, ))
        if len(items) > 1:
            raise OsbsException(
                "More than one build config found for labels: %r" %
                (label_selectors, ))

        return items[0]
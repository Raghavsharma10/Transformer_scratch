def get_build_config_by_labels_filtered(self, label_selectors, filter_key, filter_value):
        """
        Returns a build config matching the given label selectors, filtering against
        another predetermined value. This method will raise OsbsException
        if not exactly one build config is found after filtering.
        """
        items = self.get_all_build_configs_by_labels(label_selectors)

        if filter_value is not None:
            build_configs = []
            for build_config in items:
                match_value = graceful_chain_get(build_config, *filter_key.split('.'))
                if filter_value == match_value:
                    build_configs.append(build_config)
            items = build_configs

        if not items:
            raise OsbsException(
                "Build config not found for labels: %r" %
                (label_selectors, ))
        if len(items) > 1:
            raise OsbsException(
                "More than one build config found for labels: %r" %
                (label_selectors, ))
        return items[0]
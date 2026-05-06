def _compare_components(self, other, settings, ratio=False):
        """Return comparison of first, middle, and last components"""

        first = compare_name_component(
            self.first_list,
            other.first_list,
            settings['first'],
            ratio,
        )

        if settings['check_nickname']:
            if first is False:
                first = compare_name_component(
                    self.nickname_list,
                    other.first_list,
                    settings['first'],
                    ratio
                ) or compare_name_component(
                    self.first_list,
                    other.nickname_list,
                    settings['first'],
                    ratio
                )
            elif ratio and first is not 100:
                first = max(
                    compare_name_component(
                        self.nickname_list,
                        other.first_list,
                        settings['first'],
                        ratio
                    ),
                    compare_name_component(
                        self.first_list,
                        other.nickname_list,
                        settings['first'],
                        ratio
                    ),
                    first,
                )

        middle = compare_name_component(
            self.middle_list,
            other.middle_list,
            settings['middle'],
            ratio,
        )

        last = compare_name_component(
            self.last_list,
            other.last_list,
            settings['last'],
            ratio,
        )

        return first, middle, last
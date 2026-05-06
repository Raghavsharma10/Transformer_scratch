def check_edit_tab(self, tab: str, md_type: str):
        """Check if asked tab is part of Isogeo web form and reliable
        with metadata type.

        :param str tab: tab to check. Must be one one of EDIT_TABS attribute
        :param str md_type: metadata type. Must be one one of FILTER_TYPES
        """
        # check parameters types
        if not isinstance(tab, str):
            raise TypeError("'tab' expected a str value.")
        else:
            pass
        if not isinstance(md_type, str):
            raise TypeError("'md_type' expected a str value.")
        else:
            pass
        # check parameters values
        if tab not in EDIT_TABS:
            raise ValueError(
                "'{}' isn't a valid edition tab. "
                "Available values: {}".format(tab, " | ".join(EDIT_TABS))
            )
        else:
            pass
        if md_type not in FILTER_TYPES:
            if md_type in FILTER_TYPES.values():
                md_type = self._convert_md_type(md_type)
            else:
                raise ValueError(
                    "'{}' isn't a valid metadata type. "
                    "Available values: {}".format(md_type, " | ".join(FILTER_TYPES))
                )
        else:
            pass
        # check adequation tab/md_type
        if md_type not in EDIT_TABS.get(tab):
            raise ValueError(
                "'{}'  isn't a valid tab for a '{}'' metadata."
                " Only for these types: {}.".format(tab, md_type, EDIT_TABS.get(tab))
            )
        else:
            return True
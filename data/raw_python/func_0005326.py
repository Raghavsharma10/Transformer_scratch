def _check_subresource(self, subresource: str):
        """Check if specific_resources parameter is valid.

        :param str resource: subresource to check.
        """
        warnings.warn(
            "subresource in URL is deprecated." " Use _include mecanism instead.",
            DeprecationWarning,
        )
        l_subresources = (
            "conditions",
            "contacts",
            "coordinate-system",
            "events",
            "feature-attributes",
            "keywords",
            "layers",
            "limitations",
            "links",
            "operations",
            "specifications",
        )
        if isinstance(subresource, str):
            if subresource in l_subresources:
                subresource = subresource
            elif subresource == "tags":
                subresource = "keywords"
                logging.debug(
                    "'tags' is an include not a subresource."
                    " Don't worry, it has be automatically renamed "
                    "into 'keywords' which is the correct subresource."
                )
            elif subresource == "serviceLayers":
                subresource = "layers"
                logging.debug(
                    "'serviceLayers' is an include not a subresource."
                    " Don't worry, it has be automatically renamed "
                    "into 'layers' which is the correct subresource."
                )
            else:
                raise ValueError(
                    "Invalid subresource. Must be one of: {}".format(
                        "|".join(l_subresources)
                    )
                )
        else:
            raise TypeError("'subresource' expects a str")
        return subresource
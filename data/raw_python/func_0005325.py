def _check_filter_includes(self, includes: list, resource: str = "metadata"):
        """Check if specific_resources parameter is valid.

        :param list includes: sub resources to check
        :param str resource: resource type to check sub resources.
         Must be one of: metadata | keyword.
        """
        # check resource parameter
        if resource == "metadata":
            ref_subresources = _SUBRESOURCES_MD
        elif resource == "keyword":
            ref_subresources = _SUBRESOURCES_KW
        else:
            raise ValueError("Must be one of: metadata | keyword.")

        # sub resources manager
        if isinstance(includes, str) and includes.lower() == "all":
            includes = ",".join(ref_subresources)
        elif isinstance(includes, list):
            if len(includes) > 0:
                includes = ",".join(includes)
            else:
                includes = ""
        else:
            raise TypeError("'includes' expect a list or a str='all'")
        return includes
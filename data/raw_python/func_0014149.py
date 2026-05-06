def _dict_to_html_attributes(d):
        """
        Converts a dictionary to a string of ``key=\"value\"`` pairs.
        If ``None`` is provided as the dictionary an empty string is returned,
        i.e. no html attributes are generated.

        Parameters
        ----------
        d : dict
            Dictionary to convert to html attributes.

        Returns
        -------
        str
            String of HTML attributes in the form ``key_i=\"value_i\" ... key_N=\"value_N\"``,
            where ``N`` is the total number of ``(key, value)`` pairs.  
        """
        if d is None:
            return ""

        return "".join(" {}=\"{}\"".format(key, value) for key, value in iter(d.items()))
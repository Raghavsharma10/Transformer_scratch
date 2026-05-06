def get_requirement(name, requires):
        """
        Yield matching requirement strings.

        The strings are presented in the format demanded by
        pip._vendor.distlib.util.parse_requirement. Hopefully
        I'll be able to figure out a better way to handle this
        in the future. Perhaps figure out how pip does it's
        version satisfaction tests and see if it is offloadable?

        FYI there should only really be ONE matching requirement
        string, but I want to be able to process additional ones
        in case a certain package does something funky and splits
        up the requirements over multiple entries.
        """
        for require in requires:
            if name.lower() == require.project_name.lower() and require.specs:
                safe_name = require.project_name.replace('-', '_')
                yield '%s (%s)' % (safe_name, require.specifier)
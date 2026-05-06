def as_fs(self):
        """
        Returns the CPE Name as formatted string of version 2.3.

        :returns: CPE Name as formatted string
        :rtype: string
        :exception: TypeError - incompatible version
        """

        fs = []
        fs.append("cpe:2.3:")

        for i in range(0, len(CPEComponent.ordered_comp_parts)):
            ck = CPEComponent.ordered_comp_parts[i]
            lc = self._get_attribute_components(ck)

            if len(lc) > 1:
                # Incompatible version 1.1, there are two or more elements
                # in CPE Name
                errmsg = "Incompatible version {0} with formatted string".format(
                    self.VERSION)
                raise TypeError(errmsg)

            else:
                comp = lc[0]

                if (isinstance(comp, CPEComponentUndefined) or
                   isinstance(comp, CPEComponentEmpty) or
                   isinstance(comp, CPEComponentAnyValue)):

                    # Logical value any
                    v = CPEComponent2_3_FS.VALUE_ANY

                elif isinstance(comp, CPEComponentNotApplicable):

                    # Logical value not applicable
                    v = CPEComponent2_3_FS.VALUE_NA
                else:
                    # Get the value of component encoded in formatted string
                    v = comp.as_fs()

            # Append v to the formatted string then add a separator.
            fs.append(v)
            fs.append(CPEComponent2_3_FS.SEPARATOR_COMP)

        # Return the formatted string
        return CPE._trim("".join(fs[:-1]))
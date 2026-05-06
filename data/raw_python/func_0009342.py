def as_wfn(self):
        """
        Returns the CPE Name as WFN string of version 2.3.
        Only shows the first seven components.

        :return: CPE Name as WFN string
        :rtype: string
        :exception: TypeError - incompatible version
        """

        wfn = []
        wfn.append(CPE2_3_WFN.CPE_PREFIX)

        for ck in CPEComponent.CPE_COMP_KEYS:
            lc = self._get_attribute_components(ck)

            comp = lc[0]

            if (isinstance(comp, CPEComponentUndefined) or
               isinstance(comp, CPEComponentEmpty)):

                # Do not set the attribute
                continue
            else:
                v = []
                v.append(ck)
                v.append("=")

                # Get the value of WFN of component
                v.append('"')
                v.append(comp.as_wfn())
                v.append('"')

                # Append v to the WFN and add a separator
                wfn.append("".join(v))
                wfn.append(CPEComponent2_3_WFN.SEPARATOR_COMP)

        # Del the last separator
        wfn = wfn[:-1]

        # Return the WFN string
        wfn.append(CPE2_3_WFN.CPE_SUFFIX)

        return "".join(wfn)
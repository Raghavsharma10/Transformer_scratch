def as_wfn(self):
        """
        Returns the CPE Name as Well-Formed Name string of version 2.3.

        :return: CPE Name as WFN string
        :rtype: string
        :exception: TypeError - incompatible version
        """

        from .cpe2_3_wfn import CPE2_3_WFN

        wfn = []
        wfn.append(CPE2_3_WFN.CPE_PREFIX)

        for i in range(0, len(CPEComponent.ordered_comp_parts)):
            ck = CPEComponent.ordered_comp_parts[i]
            lc = self._get_attribute_components(ck)

            if len(lc) > 1:
                # Incompatible version 1.1, there are two or more elements
                # in CPE Name
                errmsg = "Incompatible version {0} with WFN".format(
                    self.VERSION)
                raise TypeError(errmsg)

            else:
                comp = lc[0]

                v = []
                v.append(ck)
                v.append("=")

                if isinstance(comp, CPEComponentAnyValue):

                    # Logical value any
                    v.append(CPEComponent2_3_WFN.VALUE_ANY)

                elif isinstance(comp, CPEComponentNotApplicable):

                    # Logical value not applicable
                    v.append(CPEComponent2_3_WFN.VALUE_NA)

                elif (isinstance(comp, CPEComponentUndefined) or
                      isinstance(comp, CPEComponentEmpty)):
                    # Do not set the attribute
                    continue
                else:
                    # Get the simple value of WFN of component
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
def as_wfn(self):
        """
        Returns the CPE Name as Well-Formed Name string of version 2.3.
        If edition component is not packed, only shows the first seven
        components, otherwise shows all.

        :return: CPE Name as WFN string
        :rtype: string
        :exception: TypeError - incompatible version
        """

        if self._str.find(CPEComponent2_3_URI.SEPARATOR_PACKED_EDITION) == -1:
            # Edition unpacked, only show the first seven components

            wfn = []
            wfn.append(CPE2_3_WFN.CPE_PREFIX)

            for ck in CPEComponent.CPE_COMP_KEYS:
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

                    if (isinstance(comp, CPEComponentUndefined) or
                       isinstance(comp, CPEComponentEmpty)):

                        # Do not set the attribute
                        continue

                    elif isinstance(comp, CPEComponentAnyValue):

                        # Logical value any
                        v.append(CPEComponent2_3_WFN.VALUE_ANY)

                    elif isinstance(comp, CPEComponentNotApplicable):

                        # Logical value not applicable
                        v.append(CPEComponent2_3_WFN.VALUE_NA)

                    else:
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

        else:
            # Shows all components
            return super(CPE2_3_URI, self).as_wfn()
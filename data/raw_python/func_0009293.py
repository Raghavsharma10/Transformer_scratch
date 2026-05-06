def _pack_edition(self):
        """
        Pack the values of the five arguments into the simple edition
        component. If all the values are blank, just return a blank.

        :returns: "edition", "sw_edition", "target_sw", "target_hw" and "other"
            attributes packed in a only value
        :rtype: string
        :exception: TypeError - incompatible version with pack operation
        """

        COMP_KEYS = (CPEComponent.ATT_EDITION,
                     CPEComponent.ATT_SW_EDITION,
                     CPEComponent.ATT_TARGET_SW,
                     CPEComponent.ATT_TARGET_HW,
                     CPEComponent.ATT_OTHER)

        separator = CPEComponent2_3_URI_edpacked.SEPARATOR_COMP

        packed_ed = []
        packed_ed.append(separator)

        for ck in COMP_KEYS:
            lc = self._get_attribute_components(ck)
            if len(lc) > 1:
                # Incompatible version 1.1, there are two or more elements
                # in CPE Name
                errmsg = "Incompatible version {0} with URI".format(
                    self.VERSION)
                raise TypeError(errmsg)

            comp = lc[0]
            if (isinstance(comp, CPEComponentUndefined) or
               isinstance(comp, CPEComponentEmpty) or
               isinstance(comp, CPEComponentAnyValue)):

                value = ""
            elif (isinstance(comp, CPEComponentNotApplicable)):
                value = CPEComponent2_3_URI.VALUE_NA
            else:
                # Component has some value; transform this original value
                # in URI value
                value = comp.as_uri_2_3()

            # Save the value of edition attribute
            if ck == CPEComponent.ATT_EDITION:
                ed = value

            # Packed the value of component
            packed_ed.append(value)
            packed_ed.append(separator)

        # Del the last separator
        packed_ed_str = "".join(packed_ed[:-1])

        only_ed = []
        only_ed.append(separator)
        only_ed.append(ed)
        only_ed.append(separator)
        only_ed.append(separator)
        only_ed.append(separator)
        only_ed.append(separator)

        only_ed_str = "".join(only_ed)

        if (packed_ed_str == only_ed_str):
            # All the extended attributes are blank,
            # so don't do any packing, just return ed
            return ed
        else:
            # Otherwise, pack the five values into a simple string
            # prefixed and internally delimited with the tilde
            return packed_ed_str
def as_uri_2_3(self):
        """
        Returns the CPE Name as URI string of version 2.3.

        :returns: CPE Name as URI string of version 2.3
        :rtype: string
        :exception: TypeError - incompatible version
        """

        uri = []
        uri.append("cpe:/")

        ordered_comp_parts = {
            0: CPEComponent.ATT_PART,
            1: CPEComponent.ATT_VENDOR,
            2: CPEComponent.ATT_PRODUCT,
            3: CPEComponent.ATT_VERSION,
            4: CPEComponent.ATT_UPDATE,
            5: CPEComponent.ATT_EDITION,
            6: CPEComponent.ATT_LANGUAGE}

        # Indicates if the previous component must be set depending on the
        # value of current component
        set_prev_comp = False
        prev_comp_list = []

        for i in range(0, len(ordered_comp_parts)):
            ck = ordered_comp_parts[i]
            lc = self._get_attribute_components(ck)

            if len(lc) > 1:
                # Incompatible version 1.1, there are two or more elements
                # in CPE Name
                errmsg = "Incompatible version {0} with URI".format(
                    self.VERSION)
                raise TypeError(errmsg)

            if ck == CPEComponent.ATT_EDITION:
                # Call the pack() helper function to compute the proper
                # binding for the edition element
                v = self._pack_edition()
                if not v:
                    set_prev_comp = True
                    prev_comp_list.append(CPEComponent2_3_URI.VALUE_ANY)
                    continue
            else:
                comp = lc[0]

                if (isinstance(comp, CPEComponentEmpty) or
                   isinstance(comp, CPEComponentAnyValue)):

                    # Logical value any
                    v = CPEComponent2_3_URI.VALUE_ANY

                elif isinstance(comp, CPEComponentNotApplicable):

                    # Logical value not applicable
                    v = CPEComponent2_3_URI.VALUE_NA
                elif isinstance(comp, CPEComponentUndefined):
                    set_prev_comp = True
                    prev_comp_list.append(CPEComponent2_3_URI.VALUE_ANY)
                    continue
                else:
                    # Get the value of component encoded in URI
                    v = comp.as_uri_2_3()

            # Append v to the URI and add a separator
            uri.append(v)
            uri.append(CPEComponent2_3_URI.SEPARATOR_COMP)

            if set_prev_comp:
                # Set the previous attribute as logical value any
                v = CPEComponent2_3_URI.VALUE_ANY
                pos_ini = max(len(uri) - len(prev_comp_list) - 1, 1)
                increment = 2  # Count of inserted values

                for p, val in enumerate(prev_comp_list):
                    pos = pos_ini + (p * increment)
                    uri.insert(pos, v)
                    uri.insert(pos + 1, CPEComponent2_3_URI.SEPARATOR_COMP)

                set_prev_comp = False
                prev_comp_list = []

        # Return the URI string, with trailing separator trimmed
        return CPE._trim("".join(uri[:-1]))
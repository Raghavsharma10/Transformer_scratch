def _parse(self):
        """
        Checks if the CPE Name is valid.

        :returns: None
        :exception: ValueError - bad-formed CPE Name
        """

        # CPE Name must not have whitespaces
        if (self._str.find(" ") != -1):
            msg = "Bad-formed CPE Name: it must not have whitespaces"
            raise ValueError(msg)

        # Partitioning of CPE Name
        parts_match = CPE2_3_FS._parts_rxc.match(self._str)

        # Validation of CPE Name parts
        if (parts_match is None):
            msg = "Bad-formed CPE Name: validation of parts failed"
            raise ValueError(msg)

        components = dict()
        parts_match_dict = parts_match.groupdict()

        for ck in CPEComponent.CPE_COMP_KEYS_EXTENDED:
            if ck in parts_match_dict:
                value = parts_match.group(ck)

                if (value == CPEComponent2_3_FS.VALUE_ANY):
                    comp = CPEComponentAnyValue()
                elif (value == CPEComponent2_3_FS.VALUE_NA):
                    comp = CPEComponentNotApplicable()
                else:
                    try:
                        comp = CPEComponent2_3_FS(value, ck)
                    except ValueError:
                        errmsg = "Bad-formed CPE Name: not correct value: {0}".format(
                            value)
                        raise ValueError(errmsg)
            else:
                errmsg = "Component {0} should be specified".format(ck)
                raise ValueError(ck)

            components[ck] = comp

        # #######################
        #  Storage of CPE Name  #
        # #######################

        part_comp = components[CPEComponent.ATT_PART]
        if isinstance(part_comp, CPEComponentLogical):
            elements = []
            elements.append(components)
            self[CPE.KEY_UNDEFINED] = elements
        else:
            # Create internal structure of CPE Name in parts:
            # one of them is filled with identified components,
            # the rest are empty
            system = parts_match.group(CPEComponent.ATT_PART)
            if system in CPEComponent.SYSTEM_VALUES:
                self._create_cpe_parts(system, components)
            else:
                self._create_cpe_parts(CPEComponent.VALUE_PART_UNDEFINED,
                                       components)

        # Fills the empty parts of internal structure of CPE Name
        for pk in CPE.CPE_PART_KEYS:
            if pk not in self.keys():
                # Empty part
                self[pk] = []
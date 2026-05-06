def _parse(self):
        """
        Checks if CPE Name is valid.

        :returns: None
        :exception: ValueError - bad-formed CPE Name
        """

        # CPE Name must not have whitespaces
        if (self._str.find(" ") != -1):
            msg = "Bad-formed CPE Name: it must not have whitespaces"
            raise ValueError(msg)

        # Partitioning of CPE Name
        parts_match = CPE2_2._parts_rxc.match(self._str)

        # Validation of CPE Name parts
        if (parts_match is None):
            msg = "Bad-formed CPE Name: validation of parts failed"
            raise ValueError(msg)

        components = dict()
        parts_match_dict = parts_match.groupdict()

        for ck in CPEComponent.CPE_COMP_KEYS:
            if ck in parts_match_dict:
                value = parts_match.group(ck)

                if (value == CPEComponent2_2.VALUE_UNDEFINED):
                    comp = CPEComponentUndefined()
                elif (value == CPEComponent2_2.VALUE_EMPTY):
                    comp = CPEComponentEmpty()
                else:
                    try:
                        comp = CPEComponent2_2(value, ck)
                    except ValueError:
                        errmsg = "Bad-formed CPE Name: not correct value: {0}".format(
                            value)
                        raise ValueError(errmsg)
            else:
                # Component not exist in this version of CPE
                comp = CPEComponentUndefined()

            components[ck] = comp

        # Adds the components of version 2.3 of CPE not defined in version 2.2
        for ck2 in CPEComponent.CPE_COMP_KEYS_EXTENDED:
            if ck2 not in components.keys():
                components[ck2] = CPEComponentUndefined()

        # #######################
        #  Storage of CPE Name  #
        # #######################

        # If part component is undefined, store it in the part without name
        if components[CPEComponent.ATT_PART] == CPEComponentUndefined():
            system = CPEComponent.VALUE_PART_UNDEFINED
        else:
            system = parts_match.group(CPEComponent.ATT_PART)

        self._create_cpe_parts(system, components)

        # Adds the undefined parts
        for sys in CPEComponent.SYSTEM_VALUES:
            if sys != system:
                pk = CPE._system_and_parts[sys]
                self[pk] = []
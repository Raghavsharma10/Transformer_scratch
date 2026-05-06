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
        parts_match = CPE2_3_URI._parts_rxc.match(self._str)

        # Validation of CPE Name parts
        if (parts_match is None):
            msg = "Bad-formed CPE Name: validation of parts failed"
            raise ValueError(msg)

        components = dict()
        edition_parts = dict()

        for ck in CPEComponent.CPE_COMP_KEYS:
            value = parts_match.group(ck)

            try:
                if (ck == CPEComponent.ATT_EDITION and value is not None):
                    if value[0] == CPEComponent2_3_URI.SEPARATOR_PACKED_EDITION:
                        # Unpack the edition part
                        edition_parts = CPE2_3_URI._unpack_edition(value)
                    else:
                        comp = CPE2_3_URI._create_component(ck, value)
                else:
                    comp = CPE2_3_URI._create_component(ck, value)
            except ValueError:
                errmsg = "Bad-formed CPE Name: not correct value '{0}'".format(
                    value)
                raise ValueError(errmsg)
            else:
                components[ck] = comp

        components = dict(components, **edition_parts)

        # Adds the components of version 2.3 of CPE not defined in version 2.2
        for ck2 in CPEComponent.CPE_COMP_KEYS_EXTENDED:
            if ck2 not in components.keys():
                components[ck2] = CPEComponentUndefined()

        # Exchange the undefined values in middle attributes of CPE Name for
        # logical value ANY
        check_change = True

        # Start in the last attribute specififed in CPE Name
        for ck in CPEComponent.CPE_COMP_KEYS[::-1]:
            if ck in components:
                comp = components[ck]
                if check_change:
                    check_change = ((ck != CPEComponent.ATT_EDITION) and
                                   (comp == CPEComponentUndefined()) or
                                   (ck == CPEComponent.ATT_EDITION and
                                   (len(edition_parts) == 0)))
                elif comp == CPEComponentUndefined():
                    comp = CPEComponentAnyValue()

                components[ck] = comp

        #  Storage of CPE Name
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
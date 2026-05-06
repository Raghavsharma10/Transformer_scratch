def _parse(self):
        """
        Checks if the CPE Name is valid.

        :returns: None
        :exception: ValueError - bad-formed CPE Name
        """

        # Check prefix and initial bracket of WFN
        if self._str[0:5] != CPE2_3_WFN.CPE_PREFIX:
            errmsg = "Bad-formed CPE Name: WFN prefix not found"
            raise ValueError(errmsg)

        # Check final backet
        if self._str[-1:] != "]":
            errmsg = "Bad-formed CPE Name: final bracket of WFN not found"
            raise ValueError(errmsg)

        content = self._str[5:-1]

        if content != "":
            # Dictionary with pairs attribute-value
            components = dict()

            # Split WFN in components
            list_component = content.split(CPEComponent2_3_WFN.SEPARATOR_COMP)

            # Adds the defined components
            for e in list_component:
                # Whitespace not valid in component names and values
                if e.find(" ") != -1:
                    msg = "Bad-formed CPE Name: WFN with too many whitespaces"
                    raise ValueError(msg)

                # Split pair attribute-value
                pair = e.split(CPEComponent2_3_WFN.SEPARATOR_PAIR)
                att_name = pair[0]
                att_value = pair[1]

                # Check valid attribute name
                if att_name not in CPEComponent.CPE_COMP_KEYS_EXTENDED:
                    msg = "Bad-formed CPE Name: invalid attribute name '{0}'".format(
                        att_name)
                    raise ValueError(msg)

                if att_name in components:
                    # Duplicate attribute
                    msg = "Bad-formed CPE Name: attribute '{0}' repeated".format(
                        att_name)
                    raise ValueError(msg)

                if not (att_value.startswith('"') and
                        att_value.endswith('"')):

                    # Logical value
                    strUpper = att_value.upper()
                    if strUpper == CPEComponent2_3_WFN.VALUE_ANY:
                        comp = CPEComponentAnyValue()
                    elif strUpper == CPEComponent2_3_WFN.VALUE_NA:
                        comp = CPEComponentNotApplicable()
                    else:
                        msg = "Invalid logical value '{0}'".format(att_value)
                        raise ValueError(msg)

                elif att_value.startswith('"') and att_value.endswith('"'):
                    # String value
                    comp = CPEComponent2_3_WFN(att_value, att_name)

                else:
                    # Bad value
                    msg = "Bad-formed CPE Name: invalid value '{0}'".format(
                        att_value)
                    raise ValueError(msg)

                components[att_name] = comp

            # Adds the undefined components
            for ck in CPEComponent.CPE_COMP_KEYS_EXTENDED:
                if ck not in components:
                    components[ck] = CPEComponentUndefined()

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
                part_value = part_comp.get_value()
                # Del double quotes of value
                system = part_value[1:-1]
                if system in CPEComponent.SYSTEM_VALUES:
                    self._create_cpe_parts(system, components)
                else:
                    self._create_cpe_parts(CPEComponent.VALUE_PART_UNDEFINED,
                                           components)

        # Fills the empty parts of internal structure of CPE Name
        for pk in CPE.CPE_PART_KEYS:
            if pk not in self.keys():
                self[pk] = []
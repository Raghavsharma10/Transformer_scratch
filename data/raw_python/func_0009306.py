def _parse(self):
        """
        Checks if the CPE Name is valid.

        :returns: None
        :exception: ValueError - bad-formed CPE Name
        """

        # CPE Name must not have whitespaces
        if (self.cpe_str.find(" ") != -1):
            errmsg = "Bad-formed CPE Name: it must not have whitespaces"
            raise ValueError(errmsg)

        # Partitioning of CPE Name in parts
        parts_match = CPE1_1._parts_rxc.match(self.cpe_str)

        # ################################
        #  Validation of CPE Name parts  #
        # ################################

        if (parts_match is None):
            errmsg = "Bad-formed CPE Name: not correct definition of CPE Name parts"
            raise ValueError(errmsg)

        CPE_PART_KEYS = (CPE.KEY_HW, CPE.KEY_OS, CPE.KEY_APP)

        for pk in CPE_PART_KEYS:
            # Get part content
            part = parts_match.group(pk)
            elements = []

            if (part is not None):
                # Part of CPE Name defined

                # ###############################
                #  Validation of part elements  #
                # ###############################

                # semicolon (;) is used to separate the part elements
                for part_elem in part.split(CPE1_1.ELEMENT_SEPARATOR):
                    j = 1

                    # ####################################
                    #  Validation of element components  #
                    # ####################################

                    components = dict()

                    # colon (:) is used to separate the element components
                    for elem_comp in part_elem.split(CPEComponent1_1.SEPARATOR_COMP):
                        comp_att = CPEComponent.ordered_comp_parts[j]

                        if elem_comp == CPEComponent1_1.VALUE_EMPTY:
                            comp = CPEComponentEmpty()
                        else:
                            try:
                                comp = CPEComponent1_1(elem_comp, comp_att)
                            except ValueError:
                                errmsg = "Bad-formed CPE Name: not correct value: {0}".format(
                                    elem_comp)
                                raise ValueError(errmsg)

                        # Identification of component name
                        components[comp_att] = comp

                        j += 1

                    # Adds the components of version 2.3 of CPE not defined
                    # in version 1.1
                    for idx in range(j, len(CPEComponent.ordered_comp_parts)):
                        comp_att = CPEComponent.ordered_comp_parts[idx]
                        components[comp_att] = CPEComponentUndefined()

                    # Get the type of system associated with CPE Name and
                    # store it in element as component
                    if (pk == CPE.KEY_HW):
                        components[CPEComponent.ATT_PART] = CPEComponent1_1(
                            CPEComponent.VALUE_PART_HW, CPEComponent.ATT_PART)
                    elif (pk == CPE.KEY_OS):
                        components[CPEComponent.ATT_PART] = CPEComponent1_1(
                            CPEComponent.VALUE_PART_OS, CPEComponent.ATT_PART)
                    elif (pk == CPE.KEY_APP):
                        components[CPEComponent.ATT_PART] = CPEComponent1_1(
                            CPEComponent.VALUE_PART_APP, CPEComponent.ATT_PART)

                    # Store the element identified
                    elements.append(components)

            # Store the part identified
            self[pk] = elements

        self[CPE.KEY_UNDEFINED] = []
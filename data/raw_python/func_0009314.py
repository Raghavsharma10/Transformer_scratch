def _unpack_edition(cls, value):
        """
        Unpack its elements and set the attributes in wfn accordingly.
        Parse out the five elements:

        ~ edition ~ software edition ~ target sw ~ target hw ~ other

        :param string value: Value of edition attribute
        :returns: Dictionary with parts of edition attribute
        :exception: ValueError - invalid value of edition attribute
        """

        components = value.split(CPEComponent2_3_URI.SEPARATOR_PACKED_EDITION)
        d = dict()

        ed = components[1]
        sw_ed = components[2]
        t_sw = components[3]
        t_hw = components[4]
        oth = components[5]

        ck = CPEComponent.ATT_EDITION
        d[ck] = CPE2_3_URI._create_component(ck, ed)
        ck = CPEComponent.ATT_SW_EDITION
        d[ck] = CPE2_3_URI._create_component(ck, sw_ed)
        ck = CPEComponent.ATT_TARGET_SW
        d[ck] = CPE2_3_URI._create_component(ck, t_sw)
        ck = CPEComponent.ATT_TARGET_HW
        d[ck] = CPE2_3_URI._create_component(ck, t_hw)
        ck = CPEComponent.ATT_OTHER
        d[ck] = CPE2_3_URI._create_component(ck, oth)

        return d
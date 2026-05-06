def set_value(self, comp_str, comp_att):
        """
        Set the value of component. By default, the component has a simple
        value.

        :param string comp_att: attribute associated with value of component
        :returns: None
        :exception: ValueError - incorrect value of component

        TEST:

        >>> val = 'xp!vista'
        >>> val2 = 'sp2'
        >>> att = CPEComponentSimple.ATT_VERSION
        >>> comp1 = CPEComponent1_1(val, att)
        >>> comp1.set_value(val2, att)
        >>> comp1.get_value()
        'sp2'
        """

        super(CPEComponent1_1, self).set_value(comp_str, comp_att)
        self._is_negated = comp_str.startswith('~')
def add_modifier(self, m_type=XenaModifierType.standard, **kwargs):
        """ Add modifier.

        :param m_type: modifier type - standard or extended.
        :type: xenamanager.xena_stram.ModifierType
        :return: newly created modifier.
        :rtype: xenamanager.xena_stream.XenaModifier
        """

        if m_type == XenaModifierType.standard:
            modifier = XenaModifier(self, index='{}/{}'.format(self.index, len(self.modifiers)))
        else:
            modifier = XenaXModifier(self, index='{}/{}'.format(self.index, len(self.xmodifiers)))
        modifier._create()
        modifier.get()
        modifier.set(**kwargs)
        return modifier
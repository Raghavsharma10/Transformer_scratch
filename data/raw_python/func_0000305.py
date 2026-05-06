def remove_modifier(self, index, m_type=XenaModifierType.standard):
        """ Remove modifier.

        :param m_type: modifier type - standard or extended.
        :param index: index of modifier to remove.
        """

        if m_type == XenaModifierType.standard:
            current_modifiers = OrderedDict(self.modifiers)
            del current_modifiers[index]

            self.set_attributes(ps_modifiercount=0)
            self.del_objects_by_type('modifier')

        else:
            current_modifiers = OrderedDict(self.xmodifiers)
            del current_modifiers[index]

            self.set_attributes(ps_modifierextcount=0)
            self.del_objects_by_type('xmodifier')

        for modifier in current_modifiers.values():
            self.add_modifier(m_type,
                              mask=modifier.mask, action=modifier.action, repeat=modifier.repeat,
                              min_val=modifier.min_val, step=modifier.step, max_val=modifier.max_val)
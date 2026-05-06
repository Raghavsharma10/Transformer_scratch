def xmodifiers(self):
        """
        :return: dictionary {index: object} of extended modifiers.
        """
        if not self.get_objects_by_type('xmodifier'):
            try:
                for index in range(int(self.get_attribute('ps_modifierextcount'))):
                    XenaXModifier(self, index='{}/{}'.format(self.index, index)).get()
            except Exception as _:
                pass
        return {s.id: s for s in self.get_objects_by_type('xmodifier')}
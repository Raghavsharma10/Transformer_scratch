def modifiers(self):
        """
        :return: dictionary {index: object} of standard modifiers.
        """
        if not self.get_objects_by_type('modifier'):
            for index in range(int(self.get_attribute('ps_modifiercount'))):
                XenaModifier(self, index='{}/{}'.format(self.index, index)).get()
        return {s.id: s for s in self.get_objects_by_type('modifier')}
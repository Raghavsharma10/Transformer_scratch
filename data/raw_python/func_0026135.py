def entryCheck(self, event = None, repair = True):
        """ Ensure any INDEF entry is uppercase, before base class behavior """
        valupr = self.choice.get().upper()
        if valupr.strip() == 'INDEF':
            self.choice.set(valupr)
        return EparOption.entryCheck(self, event, repair = repair)
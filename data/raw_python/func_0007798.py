def immediateAspects(self, ID, aspList):
        """ Returns the last separation and next application
        considering a list of possible aspects.

        """
        asps = self.aspectsByCat(ID, aspList)

        applications = asps[const.APPLICATIVE]
        separations = asps[const.SEPARATIVE]
        exact = asps[const.EXACT]

        # Get applications and separations sorted by orb

        applications = applications + [val for val in exact if val['orb'] >= 0]

        applications = sorted(applications, key=lambda var: var['orb'])
        separations = sorted(separations, key=lambda var: var['orb'])

        return (
            separations[0] if separations else None,
            applications[0] if applications else None
        )
def transitingPlanets(self):
        """ Returns a list of transiting planet objects
        """

        transitingPlanets = []

        for planet in self.planets:
            try:
                if planet.isTransiting:
                    transitingPlanets.append(planet)
            except KeyError:  # No 'discoverymethod' tag - this also filters Solar System planets
                pass

        return transitingPlanets
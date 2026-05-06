def isDiurnal(self):
        """ Returns true if this chart is diurnal. """
        sun = self.getObject(const.SUN)
        mc = self.getAngle(const.MC)
        
        # Get ecliptical positions and check if the
        # sun is above the horizon.
        lat = self.pos.lat
        sunRA, sunDecl = utils.eqCoords(sun.lon, sun.lat)
        mcRA, mcDecl = utils.eqCoords(mc.lon, 0)
        return utils.isAboveHorizon(sunRA, sunDecl, mcRA, lat)
def tiltFactor(self, midpointdepth=None,
                   printAvAngle=False):
        '''
        get tilt factor from inverse distance law
        https://en.wikipedia.org/wiki/Inverse-square_law
        '''
        # TODO: can also be only def. with FOV, rot, tilt
        beta2 = self.viewAngle(midpointdepth=midpointdepth)
        try:
            angles, vals = getattr(
                emissivity_vs_angle, self.opts['material'])()
        except AttributeError:
            raise AttributeError("material[%s] is not in list of know materials: %s" % (
                self.opts['material'], [o[0] for o in getmembers(emissivity_vs_angle)
                                        if isfunction(o[1])]))
        if printAvAngle:
            avg_angle = beta2[self.foreground()].mean()
            print('angle: %s DEG' % np.degrees(avg_angle))

        # use averaged angle instead of beta2 to not overemphasize correction
        normEmissivity = np.clip(
            InterpolatedUnivariateSpline(
                np.radians(angles), vals)(beta2), 0, 1)
        return normEmissivity
def plot_info(self, dvs):
        '''
        Plots miscellaneous de-trending information on the data
        validation summary figure.

        :param dvs: A :py:class:`dvs.DVS` figure instance

        '''

        axl, axc, axr = dvs.title()
        axc.annotate("%s %d" % (self._mission.IDSTRING, self.ID),
                     xy=(0.5, 0.5), xycoords='axes fraction',
                     ha='center', va='center', fontsize=18)

        axc.annotate(r"%.2f ppm $\rightarrow$ %.2f ppm" %
                     (self.cdppr, self.cdpp),
                     xy=(0.5, 0.2), xycoords='axes fraction',
                     ha='center', va='center', fontsize=8, color='k',
                     fontstyle='italic')

        axl.annotate("%s %s%02d: %s" %
                     (self.mission.upper(),
                      self._mission.SEASONCHAR, self.season, self.name),
                     xy=(0.5, 0.5), xycoords='axes fraction',
                     ha='center', va='center', fontsize=12,
                     color='k')

        axl.annotate(self.aperture_name if len(self.neighbors) == 0
                     else "%s, %d neighbors" %
                     (self.aperture_name, len(self.neighbors)),
                     xy=(0.5, 0.2), xycoords='axes fraction',
                     ha='center', va='center', fontsize=8, color='k',
                     fontstyle='italic')

        axr.annotate("%s %.3f" % (self._mission.MAGSTRING, self.mag),
                     xy=(0.5, 0.5), xycoords='axes fraction',
                     ha='center', va='center', fontsize=12,
                     color='k')

        if not np.isnan(self.cdppg) and self.cdppg > 0:
            axr.annotate(r"GP %.3f ppm" % (self.cdppg),
                         xy=(0.5, 0.2), xycoords='axes fraction',
                         ha='center', va='center', fontsize=8, color='k',
                         fontstyle='italic')
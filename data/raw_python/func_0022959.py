def _mpl_ax_to(self, mplobj, output='vb'):
        """Helper to get the parent axes of a given mplobj"""
        for ax in self._axs.values():
            if ax['ax'] is mplobj.axes:
                return ax[output]
        raise RuntimeError('Parent axes could not be found!')
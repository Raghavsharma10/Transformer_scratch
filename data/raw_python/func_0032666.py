def plot_campaign_outline(self, campaign=0, facecolor="#666666", text=None, dashed=False):
        """Plot the outline of a campaign as a contiguous gray patch.

        Parameters
        ----------
        campaign : int
            K2 Campaign number.

        facecolor : str
            Color of the patch.
        """
        try:
            from astropy.coordinates import SkyCoord
        except ImportError:
            logger.error("You need to install AstroPy for this feature.")
            return None
        # The outline is composed of two filled rectangles,
        # defined by the first coordinate of the corner of four channels each
        fov = getKeplerFov(campaign)
        corners = fov.getCoordsOfChannelCorners()
        for rectangle in [[4, 75, 84, 11], [15, 56, 71, 32]]:
            ra_outline, dec_outline = [], []
            for channel in rectangle:
                idx = np.where(corners[::, 2] == channel)
                ra_outline.append(corners[idx, 3][0][0])
                dec_outline.append(corners[idx, 4][0][0])

            crd = SkyCoord(ra_outline, dec_outline, unit='deg')
            l = crd.galactic.l.deg
            if campaign not in [4, 13, 1713]:
                l[l > 180] -= 360
            l, b = list(l), list(crd.galactic.b.deg)
            if dashed:
                myfill = self.ax.fill(l + l[:1],
                                      b + b[:1],
                                      facecolor=facecolor, zorder=151, lw=2, ls='dashed',
                                      edgecolor='white')
                # myfill = self.ax.plot(l + l[:1],
                #                      b + b[:1],
                #                      zorder=200, lw=2,
                #                      ls='dotted', color='white')
            else:
                myfill = self.ax.fill(l + l[:1],
                                      b + b[:1],
                                      facecolor=facecolor, zorder=151, lw=0)
        # Print the campaign number on top of the outline
        ra, dec, roll = fov.getBoresight()
        gal = SkyCoord(ra, dec, unit='deg').galactic
        l, b = gal.l.deg, gal.b.deg
        if l > 180:
            l -= 360
        if text is None:
            text = "{}".format(campaign)
        self.ax.text(l, b, text,
                     fontsize=14, color="white", ha="center", va="center",
                     zorder=255)
        return myfill
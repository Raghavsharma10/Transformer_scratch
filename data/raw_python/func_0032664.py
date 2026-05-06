def plot_campaign_outline(self, campaign=0, facecolor="#666666", text=None):
        """Plot the outline of a campaign as a contiguous gray patch.

        Parameters
        ----------
        campaign : int
            K2 Campaign number.

        facecolor : str
            Color of the patch.
        """
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
            ra = np.array(ra_outline + ra_outline[:1])
            dec = np.array(dec_outline + dec_outline[:1])
            if campaign == 1002:  # Overlaps the meridian
                ra[ra > 180] -= 360
            myfill = self.ax.fill(ra, dec,
                                  facecolor=facecolor,
                                  zorder=151, lw=0)
        # Print the campaign number on top of the outline
        if text is None:
            text = "{}".format(campaign)
        ra_center, dec_center, _ = fov.getBoresight()
        if campaign == 6:
            dec_center -= 2
        elif campaign == 12:
            ra_center += 0.5
            dec_center -= 1.7
        elif campaign == 13:
            dec_center -= 1.5
        elif campaign == 16:
            dec_center += 1.5
        elif campaign == 18:
            dec_center -= 1.5
        elif campaign == 19:
            dec_center += 1.7
        elif campaign == 20:
            dec_center += 1.5
        offsets = {5: (40, -20), 16: (-20, 40), 18: (-15, -50)}
        if campaign in [5]:
            pl.annotate(text, xy=(ra_center, dec_center),
                        xycoords='data', ha='center',
                        xytext=offsets[campaign], textcoords='offset points',
                        size=18, zorder=0, color=facecolor,
                        arrowprops=dict(arrowstyle="-", ec=facecolor, lw=2))
        else:
            self.ax.text(ra_center, dec_center, text,
                         fontsize=18, color="white",
                         ha="center", va="center",
                         zorder=155)
        return myfill
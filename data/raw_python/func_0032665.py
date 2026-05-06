def plot_campaign(self, campaign=0, annotate_channels=True, **kwargs):
        """Plot all the active channels of a campaign."""
        fov = getKeplerFov(campaign)
        corners = fov.getCoordsOfChannelCorners()

        for ch in np.arange(1, 85, dtype=int):
            if ch in fov.brokenChannels:
                continue  # certain channel are no longer used
            idx = np.where(corners[::, 2] == ch)
            mdl = int(corners[idx, 0][0][0])
            out = int(corners[idx, 1][0][0])
            ra = corners[idx, 3][0]
            if campaign == 1002:  # Concept Engineering Test overlapped the meridian
                ra[ra < 180] += 360
            dec = corners[idx, 4][0]
            self.ax.fill(np.concatenate((ra, ra[:1])),
                         np.concatenate((dec, dec[:1])), **kwargs)
            if annotate_channels:
                txt = "K2C{0}\n{1}.{2}\n#{3}".format(campaign, mdl, out, ch)
                txt = "{1}.{2}\n#{3}".format(campaign, mdl, out, ch)
                self.ax.text(np.mean(ra), np.mean(dec), txt,
                             ha="center", va="center",
                             zorder=91, fontsize=10,
                             color="#000000", clip_on=True)
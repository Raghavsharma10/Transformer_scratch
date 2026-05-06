def plot_outline(self, annotate_late_targets=False, annotate_channels=False):
        """Plots the coverage of both the channels and the C9 superstamp."""
        fov = getKeplerFov(9)
        # Plot the superstamp
        superstamp_patches = []
        for ch in SUPERSTAMP["channels"]:
            v_col = SUPERSTAMP["channels"][ch]["vertices_col"]
            v_row = SUPERSTAMP["channels"][ch]["vertices_row"]
            radec = np.array([
                                fov.getRaDecForChannelColRow(int(ch),
                                                             v_col[idx],
                                                             v_row[idx])
                                for idx in range(len(v_col))
                              ])
            patch = self.ax.fill(radec[:, 0], radec[:, 1],
                                 lw=0, facecolor="#27ae60", zorder=100)
            superstamp_patches.append(patch)

        # Plot the late target masks
        late_target_patches = []
        for mask in LATE_TARGETS["masks"]:
            ch = mask["channel"]
            v_col = mask["vertices_col"]
            v_row = mask["vertices_row"]
            radec = np.array([
                                fov.getRaDecForChannelColRow(int(ch),
                                                             v_col[idx],
                                                             v_row[idx])
                                for idx in range(len(v_col))
                              ])
            patch = self.ax.fill(radec[:, 0], radec[:, 1],
                                 lw=0, facecolor="#27ae60", zorder=201)
            late_target_patches.append(patch)
            if annotate_late_targets and 'context' not in mask["name"]:
                self.ax.text(np.mean(radec[:, 0]), np.mean(radec[:, 1]), '  ' + mask["name"],
                             ha="left", va="center",
                             zorder=950, fontsize=10,
                             color="#c0392b", clip_on=True)

        # Plot all channel outlines
        channel_patches = []
        corners = fov.getCoordsOfChannelCorners()
        for ch in np.arange(1, 85, dtype=int):
            if ch in fov.brokenChannels:
                continue  # certain channel are no longer used
            idx = np.where(corners[::, 2] == ch)
            mdl = int(corners[idx, 0][0][0])
            out = int(corners[idx, 1][0][0])
            ra = corners[idx, 3][0]
            dec = corners[idx, 4][0]
            patch = self.ax.fill(np.concatenate((ra, ra[:1])),
                                 np.concatenate((dec, dec[:1])),
                                 lw=0, facecolor="#cccccc", zorder=50)
            channel_patches.append(patch)
            if annotate_channels:
                txt = "{}.{}\n#{}".format(mdl, out, ch)
                self.ax.text(np.mean(ra), np.mean(dec), txt,
                             ha="center", va="center",
                             zorder=900, fontsize=14,
                             color="#000000", clip_on=True)
        return superstamp_patches, channel_patches
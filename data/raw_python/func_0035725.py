def plot(self, style='heatmap', legend=False, cmap=None, ax=None):
        """
        Presents the AstonFrame using matplotlib.

        Parameters
        ----------
        style : {'heatmap', 'colors', ''}
        legend : bool, optional
        cmap: matplotlib.colors.Colormap, optional
        ax : matplotlib.axes.Axes, optional

        """
        # styles: 2d, colors, otherwise interpret as trace?
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        if style == 'heatmap':
            ions = self.columns
            ext = (self.index[0], self.index[-1], min(ions), max(ions))
            grid = self.values[:, np.argsort(self.columns)].transpose()
            if isinstance(self.values, scipy.sparse.spmatrix):
                grid = grid.toarray()
            img = ax.imshow(grid, origin='lower', aspect='auto',
                            extent=ext, cmap=cmap)
            if legend:
                ax.figure.colorbar(img)
        elif style == 'colors':
            # TODO: importing gaussian at the top leads to a whole
            # mess of dependency issues => fix somehow?
            from aston.peak.peak_models import gaussian
            from matplotlib.colors import ListedColormap

            wvs = np.genfromtxt(np.array(self.columns).astype(bytes))
            # wvs = self.columns.astype(float)

            # http://www.ppsloan.org/publications/XYZJCGT.pdf
            vis_filt = np.zeros((3, len(wvs)))
            vis_filt[0] = 1.065 * gaussian(wvs, x=595.8, w=33.33) + \
                0.366 * gaussian(wvs, x=446.8, w=19.44)
            vis_filt[1] = 1.014 * gaussian(np.log(wvs), x=np.log(556.3),
                                           w=0.075)
            vis_filt[2] = 1.839 * gaussian(np.log(wvs), x=np.log(449.8),
                                           w=0.051)
            if isinstance(self.values, scipy.sparse.spmatrix):
                xyz = np.dot(self.values.toarray(), vis_filt.T)
            else:
                xyz = np.dot(self.values.copy(), vis_filt.T)

            # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
            xyz_rgb = [[3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [0.0556434, -0.2040259, 1.0572252]]
            xyz_rgb = np.array(xyz_rgb)
            rgb = np.dot(xyz_rgb, xyz.T).T

            # normalize
            rgb[rgb < 0] = 0
            rgb /= np.max(rgb)
            rgb = 1 - np.abs(rgb)

            # plot
            cmask = np.meshgrid(np.arange(rgb.shape[0]), 0)[0]
            ax.imshow(cmask, cmap=ListedColormap(rgb), aspect='auto',
                      extent=(self.index[0], self.index[-1], 0, 1))
            ax.yaxis.set_ticks([])
        else:
            if cmap is not None:
                color = cmap(0, 1)
            else:
                color = 'k'
            self.trace().plot(color=color, ax=ax)
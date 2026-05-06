def _plot_connectivity_helper(self, ii, ji, mat_datai, data, lims=[1, 8]):
        """
        A debug function used to plot the adjacency/connectivity matrix.
        """
        from matplotlib.pyplot import quiver, colorbar, clim,  matshow
        I = ~np.isnan(mat_datai) & (ji != -1) & (mat_datai >= 0)
        mat_data = mat_datai[I]
        j = ji[I]
        i = ii[I]
        x = i.astype(float) % data.shape[1]
        y = i.astype(float) // data.shape[1]
        x1 = (j.astype(float) % data.shape[1]).ravel()
        y1 = (j.astype(float) // data.shape[1]).ravel()
        nx = (x1 - x)
        ny = (y1 - y)
        matshow(data, cmap='gist_rainbow'); colorbar(); clim(lims)
        quiver(x, y, nx, ny, mat_data.ravel(), angles='xy', scale_units='xy',
               scale=1, cmap='bone')
        colorbar(); clim([0, 1])
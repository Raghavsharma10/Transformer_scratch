def imshow(self, *args, show_crosshair=True, show_mask=True,
               show_qscale=True, axes=None, invalid_color='black',
               mask_opacity=0.8, show_colorbar=True, **kwargs):
        """Plot the matrix (imshow)

        Keyword arguments [and their default values]:

        show_crosshair [True]: if a cross-hair marking the beam position is
            to be plotted.
        show_mask [True]: if the mask is to be plotted.
        show_qscale [True]: if the horizontal and vertical axes are to be
            scaled into q
        axes [None]: the axes into which the image should be plotted. If
            None, defaults to the currently active axes (returned by plt.gca())
        invalid_color ['black']: the color for invalid (NaN or infinite) pixels
        mask_opacity [0.8]: the opacity of the overlaid mask (1 is fully
            opaque, 0 is fully transparent)
        show_colorbar [True]: if a colorbar is to be added. Can be a boolean
            value (True or False) or an instance of matplotlib.axes.Axes, into
            which the color bar should be drawn.

        All other keywords are forwarded to plt.imshow() or
            matplotlib.Axes.imshow()

        Returns: the image instance returned by imshow()
        """
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        if 'origin' not in kwargs:
            kwargs['origin'] = 'upper'
        if show_qscale:
            ymin, xmin = self.pixel_to_q(0, 0)
            ymax, xmax = self.pixel_to_q(*self.shape)
            if kwargs['origin'].upper() == 'UPPER':
                kwargs['extent'] = [xmin, xmax, -ymax, -ymin]
            else:
                kwargs['extent'] = [xmin, xmax, ymin, ymax]
            bcx = 0
            bcy = 0
        else:
            bcx = self.header.beamcenterx
            bcy = self.header.beamcentery
            xmin = 0
            xmax = self.shape[1]
            ymin = 0
            ymax = self.shape[0]
            if kwargs['origin'].upper() == 'UPPER':
                kwargs['extent'] = [0, self.shape[1], self.shape[0], 0]
            else:
                kwargs['extent'] = [0, self.shape[1], 0, self.shape[0]]
        if axes is None:
            axes = plt.gca()
        ret = axes.imshow(self.intensity, **kwargs)
        if show_mask:
            # workaround: because of the colour-scaling we do here, full one and
            #   full zero masks look the SAME, i.e. all the image is shaded.
            #   Thus if we have a fully unmasked matrix, skip this section.
            #   This also conserves memory.
            if (self.mask == 0).sum():  # there are some masked pixels
                # we construct another representation of the mask, where the masked pixels are 1.0, and the
                # unmasked ones will be np.nan. They will thus be not rendered.
                mf = np.ones(self.mask.shape, np.float)
                mf[self.mask != 0] = np.nan
                kwargs['cmap'] = matplotlib.cm.gray_r
                kwargs['alpha'] = mask_opacity
                kwargs['norm'] = matplotlib.colors.Normalize()
                axes.imshow(mf, **kwargs)
        if show_crosshair:
            ax = axes.axis()  # save zoom state
            axes.plot([xmin, xmax], [bcy] * 2, 'w-')
            axes.plot([bcx] * 2, [ymin, ymax], 'w-')
            axes.axis(ax)  # restore zoom state
        axes.set_facecolor(invalid_color)
        if show_colorbar:
            if isinstance(show_colorbar, matplotlib.axes.Axes):
                axes.figure.colorbar(
                        ret, cax=show_colorbar)
            else:
                # try to find a suitable colorbar axes: check if the plot target axes already
                # contains some images, then check if their colorbars exist as
                # axes.
                cax = [i.colorbar[1]
                       for i in axes.images if i.colorbar is not None]
                cax = [c for c in cax if c in c.figure.axes]
                if cax:
                    cax = cax[0]
                else:
                    cax = None
                axes.figure.colorbar(ret, cax=cax, ax=axes)
        axes.figure.canvas.draw()
        return ret
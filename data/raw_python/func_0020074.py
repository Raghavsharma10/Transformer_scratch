def plot_aperture(self, axes, labelsize=8):
        '''
        Plots the aperture and the pixel images at the beginning, middle,
        and end of the time series. Also plots a high resolution image of
        the target, if available.

        '''

        log.info('Plotting the aperture...')

        # Get colormap
        plasma = pl.get_cmap('plasma')
        plasma.set_bad(alpha=0)

        # Get aperture contour
        def PadWithZeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        ny, nx = self.pixel_images[0].shape
        contour = np.zeros((ny, nx))
        contour[np.where(self.aperture)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode='nearest')
        extent = np.array([-1, nx, -1, ny])

        # Plot first, mid, and last TPF image
        title = ['start', 'mid', 'end']
        for i, image in enumerate(self.pixel_images):
            ax = axes[i]
            ax.imshow(image, aspect='auto',
                      interpolation='nearest', cmap=plasma)
            ax.contour(highres, levels=[0.5], extent=extent,
                       origin='lower', colors='r', linewidths=1)

            # Check for saturated columns
            for x in range(self.aperture.shape[0]):
                for y in range(self.aperture.shape[1]):
                    if self.aperture[x][y] == AP_SATURATED_PIXEL:
                        ax.fill([y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                                [x - 0.5, x - 0.5, x + 0.5, x + 0.5],
                                fill=False, hatch='xxxxx', color='r', lw=0)

            ax.axis('off')
            ax.set_xlim(-0.7, nx - 0.3)
            ax.set_ylim(-0.7, ny - 0.3)
            ax.annotate(title[i], xy=(0.5, 0.975), xycoords='axes fraction',
                        ha='center', va='top', size=labelsize, color='w')
            if i == 1:
                for source in self.nearby:
                    ax.annotate('%.1f' % source['mag'],
                                xy=(source['x'] - source['x0'],
                                    source['y'] - source['y0']),
                                ha='center', va='center', size=labelsize - 2,
                                color='w', fontweight='bold')

        # Plot hi res image
        if self.hires is not None:
            ax = axes[-1]
            ax.imshow(self.hires, aspect='auto',
                      extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
                      interpolation='bicubic', cmap=plasma)
            ax.contour(highres, levels=[0.5], extent=extent,
                       origin='lower', colors='r', linewidths=1)
            ax.axis('off')
            ax.set_xlim(-0.7, nx - 0.3)
            ax.set_ylim(-0.7, ny - 0.3)
            ax.annotate('hires', xy=(0.5, 0.975), xycoords='axes fraction',
                        ha='center', va='top', size=labelsize, color='w')
        else:
            ax = axes[-1]
            ax.axis('off')
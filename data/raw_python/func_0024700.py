def _do_plot(x, y, title='', xlog=False, ylog=False,
                 left=None, right=None, bottom=None, top=None,
                 save_as=''):  # pragma: no cover
        """Plot worker.

        Parameters
        ----------
        x, y : `~astropy.units.quantity.Quantity`
            Wavelength and flux/throughput to plot.

        kwargs
            See :func:`plot`.

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.error('No matplotlib installation found; plotting disabled '
                      'as a result.')
            return

        fig, ax = plt.subplots()
        ax.plot(x, y)

        # Custom wavelength limits
        if left is not None:
            ax.set_xlim(left=left)
        if right is not None:
            ax.set_xlim(right=right)

        # Custom flux/throughput limit
        if bottom is not None:
            ax.set_ylim(bottom=bottom)
        if top is not None:
            ax.set_ylim(top=top)

        xu = x.unit
        if xu.physical_type == 'frequency':
            ax.set_xlabel('Frequency ({0})'.format(xu))
        else:
            ax.set_xlabel('Wavelength ({0})'.format(xu))

        yu = y.unit
        if yu is u.dimensionless_unscaled:
            ax.set_ylabel('Unitless')
        else:
            ax.set_ylabel('Flux ({0})'.format(yu))

        if title:
            ax.set_title(title)

        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')

        plt.draw()

        if save_as:
            plt.savefig(save_as)
            log.info('Plot saved as {0}'.format(save_as))
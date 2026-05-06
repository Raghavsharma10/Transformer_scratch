def plot_contpix(self, x, y, contpix_x, contpix_y, figname):
        """ Plot baseline spec with continuum pix overlaid 

        Parameters
        ----------
        """
        fig, axarr = plt.subplots(2, sharex=True)
        plt.xlabel(r"Wavelength $\lambda (\AA)$")
        plt.xlim(min(x), max(x))
        ax = axarr[0]
        ax.step(x, y, where='mid', c='k', linewidth=0.3,
                label=r'$\theta_0$' + "= the leading fit coefficient")
        ax.scatter(contpix_x, contpix_y, s=1, color='r',
                label="continuum pixels")
        ax.legend(loc='lower right', 
                prop={'family':'serif', 'size':'small'})
        ax.set_title("Baseline Spectrum with Continuum Pixels")
        ax.set_ylabel(r'$\theta_0$')
        ax = axarr[1]
        ax.step(x, y, where='mid', c='k', linewidth=0.3,
             label=r'$\theta_0$' + "= the leading fit coefficient")
        ax.scatter(contpix_x, contpix_y, s=1, color='r',
                label="continuum pixels")
        ax.set_title("Baseline Spectrum with Continuum Pixels, Zoomed")
        ax.legend(loc='upper right', prop={'family':'serif', 
            'size':'small'})
        ax.set_ylabel(r'$\theta_0$')
        ax.set_ylim(0.95, 1.05)
        print("Diagnostic plot: fitted 0th order spec w/ cont pix")
        print("Saved as %s.png" % (figname))
        plt.savefig(figname)
        plt.close()
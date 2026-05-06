def overlay_spectra(model, dataset):
    """ Run a series of diagnostics on the fitted spectra 

    Parameters
    ----------
    model: model
        best-fit Cannon spectral model
    
    dataset: Dataset
        original spectra

    """
    best_flux, best_ivar = draw_spectra(model, dataset)
    coeffs_all, covs, scatters, all_chisqs, pivots, label_vector = model.model

    # Overplot original spectra with best-fit spectra
    print("Overplotting spectra for ten random stars")
    res = dataset.test_flux-best_flux
    lambdas = dataset.wl
    npix = len(lambdas)
    nstars = best_flux.shape[0]
    pickstars = []
    for i in range(10):
        pickstars.append(random.randrange(0, nstars-1))
    for i in pickstars:
        print("Star %s" % i)
        ID = dataset.test_ID[i]
        spec_orig = dataset.test_flux[i,:]
        bad = dataset.test_flux[i,:] == 0
        lambdas = np.ma.array(lambdas, mask=bad, dtype=float)
        npix = len(lambdas.compressed())
        spec_orig = np.ma.array(dataset.test_flux[i,:], mask=bad)
        spec_fit = np.ma.array(best_flux[i,:], mask=bad)
        ivars_orig = np.ma.array(dataset.test_ivar[i,:], mask=bad)
        ivars_fit = np.ma.array(best_ivar[i,:], mask=bad)
        red_chisq = np.sum(all_chisqs[:,i], axis=0) / (npix - coeffs_all.shape[1])
        red_chisq = np.round(red_chisq, 2)
        fig,axarr = plt.subplots(2)
        ax1 = axarr[0]
        im = ax1.scatter(lambdas, spec_orig, label="Orig Spec",
                         c=1 / np.sqrt(ivars_orig), s=10)
        ax1.scatter(lambdas, spec_fit, label="Cannon Spec", c='r', s=10)
        ax1.errorbar(lambdas, spec_fit, 
                     yerr=1/np.sqrt(ivars_fit), fmt='ro', ms=1, alpha=0.7)
        ax1.set_xlabel(r"Wavelength $\lambda (\AA)$")
        ax1.set_ylabel("Normalized flux")
        ax1.set_title("Spectrum Fit: %s" % ID)
        ax1.set_title("Spectrum Fit")
        ax1.set_xlim(min(lambdas.compressed())-10, max(lambdas.compressed())+10)
        ax1.legend(loc='lower center', fancybox=True, shadow=True)
        ax2 = axarr[1]
        ax2.scatter(spec_orig, spec_fit, c=1/np.sqrt(ivars_orig), alpha=0.7)
        ax2.errorbar(spec_orig, spec_fit, yerr=1 / np.sqrt(ivars_fit),
                     ecolor='k', fmt="none", ms=1, alpha=0.7)
        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar()
        #fig.colorbar(
        #        im, cax=cbar_ax,
        #        label="Uncertainties on the Fluxes from the Original Spectrum")
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        lims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
        ax2.plot(lims, lims, 'k-', alpha=0.75)
        textstr = "Red Chi Sq: %s" % red_chisq
        props = dict(boxstyle='round', facecolor='palevioletred', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.set_xlabel("Orig Fluxes")
        ax2.set_ylabel("Fitted Fluxes")
        plt.tight_layout()
        filename = "best_fit_spec_Star%s.png" % i
        print("Saved as %s" % filename)
        fig.savefig(filename)
        plt.close(fig)
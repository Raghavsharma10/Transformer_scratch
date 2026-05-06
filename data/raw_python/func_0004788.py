def diagnostics_plot_chisq(self, ds, figname = "modelfit_chisqs.png"):
        """ Produce a set of diagnostic plots for the model 

        Parameters
        ----------
        (optional) chisq_dist_plot_name: str
            Filename of output saved plot
        """
        label_names = ds.get_plotting_labels()
        lams = ds.wl
        pivots = self.pivots
        npixels = len(lams)
        nlabels = len(pivots)
        chisqs = self.chisqs
        coeffs = self.coeffs
        scatters = self.scatters

        # Histogram of the chi squareds of ind. stars
        plt.hist(np.sum(chisqs, axis=0), color='lightblue', alpha=0.7,
                bins=int(np.sqrt(len(chisqs))))
        dof = len(lams) - coeffs.shape[1]   # for one star
        plt.axvline(x=dof, c='k', linewidth=2, label="DOF")
        plt.legend()
        plt.title("Distribution of " + r"$\chi^2$" + " of the Model Fit")
        plt.ylabel("Count")
        plt.xlabel(r"$\chi^2$" + " of Individual Star")
        print("Diagnostic plot: histogram of the red chi squareds of the fit")
        print("Saved as %s" %figname)
        plt.savefig(figname)
        plt.close()
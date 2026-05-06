def imshow(
        self,
        data=None,
        save=False,
        ax=None,
        interpolation="none",
        extra_title=None,
        show_resonances="some",
        set_extent=True,
        equalized=False,
        rmin=None,
        rmax=None,
        savepath=".",
        **kwargs,
    ):
        """Powerful default display.

        show_resonances can be True, a list, 'all', or 'some'
        """
        if data is None:
            data = self.img
        if self.resonance_axis is not None:
            logger.debug("removing resonance_axis")
            self.resonance_axis.remove()
        if equalized:
            data = np.nan_to_num(data)
            data[data < 0] = 0
            data = exposure.equalize_hist(data)
        self.plotted_data = data

        extent_val = self.extent if set_extent else None
        min_, max_ = self.plot_limits
        self.min_ = min_
        self.max_ = max_
        if ax is None:
            if not _SEABORN_INSTALLED:
                fig, ax = plt.subplots(figsize=calc_4_3(8))
            else:
                fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        with quantity_support():
            im = ax.imshow(
                data,
                extent=extent_val,
                cmap="gray",
                vmin=min_,
                vmax=max_,
                interpolation=interpolation,
                origin="lower",
                aspect="auto",
                **kwargs,
            )
        if any([rmin is not None, rmax is not None]):
            ax.set_ylim(rmin, rmax)
        self.mpl_im = im
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Radius [Mm]")
        ax.ticklabel_format(useOffset=False)
        # ax.grid('on')
        title = self.plot_title
        if extra_title:
            title += ", " + extra_title
        ax.set_title(title, fontsize=12)
        if show_resonances:
            self.set_resonance_axis(ax, show_resonances, rmin, rmax)
        if save:
            savename = self.plotfname
            if extra_title:
                savename = savename[:-4] + "_" + extra_title + ".png"
            p = Path(savename)
            fullpath = Path(savepath) / p.name
            fig.savefig(fullpath, dpi=150)
            logging.info("Created %s", fullpath)
        self.im = im
        return im
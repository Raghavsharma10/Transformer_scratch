def plot(self, title='TimeMoc', view=(None, None)):
        """
        Plot the TimeMoc in a time window.

        This method uses interactive matplotlib. The user can move its mouse through the plot to see the
        time (at the mouse position).

        Parameters
        ----------
        title : str, optional
            The title of the plot. Set to 'TimeMoc' by default.
        view : (`~astropy.time.Time`, `~astropy.time.Time`), optional
            Define the view window in which the observations are plotted. Set to (None, None) by default (i.e.
            all the observation time window is rendered).

        """
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt

        if self._interval_set.empty():
            print('Nothing to print. This TimeMoc object is empty.')
            return

        plot_order = 15
        if self.max_order > plot_order:
            plotted_moc = self.degrade_to_order(plot_order)
        else:
            plotted_moc = self

        min_jd = plotted_moc.min_time.jd if not view[0] else view[0].jd
        max_jd = plotted_moc.max_time.jd if not view[1] else view[1].jd

        if max_jd < min_jd:
            raise ValueError("Invalid selection: max_jd = {0} must be > to min_jd = {1}".format(max_jd, min_jd))

        fig1 = plt.figure(figsize=(9.5, 5))
        ax = fig1.add_subplot(111)

        ax.set_xlabel('iso')
        ax.get_yaxis().set_visible(False)

        size = 2000
        delta = (max_jd - min_jd) / size
        min_jd_time = min_jd

        ax.set_xticks([0, size])
        ax.set_xticklabels(Time([min_jd_time, max_jd], format='jd', scale='tdb').iso, rotation=70)

        y = np.zeros(size)
        for (s_time_us, e_time_us) in plotted_moc._interval_set._intervals:
            s_index = int((s_time_us / TimeMOC.DAY_MICRO_SEC - min_jd_time) / delta)
            e_index = int((e_time_us / TimeMOC.DAY_MICRO_SEC - min_jd_time) / delta)
            y[s_index:(e_index+1)] = 1.0

        # hack in case of full time mocs.
        if np.all(y):
            y[0] = 0

        z = np.tile(y, (int(size//10), 1))

        plt.title(title)

        color_map = LinearSegmentedColormap.from_list('w2r', ['#fffff0', '#aa0000'])
        color_map.set_under('w')
        color_map.set_bad('gray')

        plt.imshow(z, interpolation='bilinear', cmap=color_map)

        def on_mouse_motion(event):
            for txt in ax.texts:
                txt.set_visible(False)

            text = ax.text(0, 0, "", va="bottom", ha="left")

            time = Time(event.xdata * delta + min_jd_time, format='jd', scale='tdb')

            tx = '{0}'.format(time.iso)
            text.set_position((event.xdata - 50, 700))
            text.set_rotation(70)
            text.set_text(tx)

        cid = fig1.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

        plt.show()
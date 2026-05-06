def plot_lc(self, ax, info_left='', info_right='', color='b'):
        '''
        Plots the current light curve. This is called at several stages to
        plot the de-trending progress as a function of the different
        *PLD* orders.

        :param ax: The current :py:obj:`matplotlib.pyplot` axis instance
        :param str info_left: Information to display at the left of the \
               plot. Default `''`
        :param str info_right: Information to display at the right of the \
               plot. Default `''`
        :param str color: The color of the data points. Default `'b'`

        '''

        # Plot
        if (self.cadence == 'lc') or (len(self.time) < 4000):
            ax.plot(self.apply_mask(self.time), self.apply_mask(self.flux),
                    ls='none', marker='.', color=color,
                    markersize=2, alpha=0.5)
            ax.plot(self.time[self.transitmask], self.flux[self.transitmask],
                    ls='none', marker='.', color=color,
                    markersize=2, alpha=0.5)
        else:
            ax.plot(self.apply_mask(self.time), self.apply_mask(
                    self.flux), ls='none', marker='.', color=color,
                    markersize=2, alpha=0.03, zorder=-1)
            ax.plot(self.time[self.transitmask], self.flux[self.transitmask],
                    ls='none', marker='.', color=color,
                    markersize=2, alpha=0.03, zorder=-1)
            ax.set_rasterization_zorder(0)
        ylim = self.get_ylim()

        # Plot the outliers, but not the NaNs
        badmask = [i for i in self.badmask if i not in self.nanmask]

        def O1(x): return x[self.outmask]

        def O2(x): return x[badmask]
        if self.cadence == 'lc':
            ax.plot(O1(self.time), O1(self.flux), ls='none',
                    color="#777777", marker='.', markersize=2, alpha=0.5)
            ax.plot(O2(self.time), O2(self.flux),
                    'r.', markersize=2, alpha=0.25)
        else:
            ax.plot(O1(self.time), O1(self.flux), ls='none', color="#777777",
                    marker='.', markersize=2, alpha=0.25, zorder=-1)
            ax.plot(O2(self.time), O2(self.flux), 'r.',
                    markersize=2, alpha=0.125, zorder=-1)
        for i in np.where(self.flux < ylim[0])[0]:
            if i in badmask:
                color = "#ffcccc"
            elif i in self.outmask:
                color = "#cccccc"
            elif i in self.nanmask:
                continue
            else:
                color = "#ccccff"
            ax.annotate('', xy=(self.time[i], ylim[0]), xycoords='data',
                        xytext=(0, 15), textcoords='offset points',
                        arrowprops=dict(arrowstyle="-|>", color=color))
        for i in np.where(self.flux > ylim[1])[0]:
            if i in badmask:
                color = "#ffcccc"
            elif i in self.outmask:
                color = "#cccccc"
            elif i in self.nanmask:
                continue
            else:
                color = "#ccccff"
            ax.annotate('', xy=(self.time[i], ylim[1]), xycoords='data',
                        xytext=(0, -15), textcoords='offset points',
                        arrowprops=dict(arrowstyle="-|>", color=color))

        # Plot the breakpoints
        for brkpt in self.breakpoints[:-1]:
            if len(self.breakpoints) <= 5:
                ax.axvline(self.time[brkpt], color='r', ls='--', alpha=0.5)
            else:
                ax.axvline(self.time[brkpt], color='r', ls='-', alpha=0.025)

        # Appearance
        if len(self.cdpp_arr) == 2:
            ax.annotate('%.2f ppm' % self.cdpp_arr[0], xy=(0.02, 0.975),
                        xycoords='axes fraction',
                        ha='left', va='top', fontsize=10)
            ax.annotate('%.2f ppm' % self.cdpp_arr[1], xy=(0.98, 0.975),
                        xycoords='axes fraction',
                        ha='right', va='top', fontsize=10)
        elif len(self.cdpp_arr) < 6:
            for n in range(len(self.cdpp_arr)):
                if n > 0:
                    x = (self.time[self.breakpoints[n - 1]] - self.time[0]
                         ) / (self.time[-1] - self.time[0]) + 0.02
                else:
                    x = 0.02
                ax.annotate('%.2f ppm' % self.cdpp_arr[n], xy=(x, 0.975),
                            xycoords='axes fraction',
                            ha='left', va='top', fontsize=8)
        else:
            ax.annotate('%.2f ppm' % self.cdpp, xy=(0.02, 0.975),
                        xycoords='axes fraction',
                        ha='left', va='top', fontsize=10)
        ax.annotate(info_right, xy=(0.98, 0.025), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10, alpha=0.5,
                    fontweight='bold')
        ax.annotate(info_left, xy=(0.02, 0.025), xycoords='axes fraction',
                    ha='left', va='bottom', fontsize=8)
        ax.set_xlabel(r'Time (%s)' % self._mission.TIMEUNITS, fontsize=5)
        ax.margins(0.01, 0.1)
        ax.set_ylim(*ylim)
        ax.get_yaxis().set_major_formatter(Formatter.Flux)
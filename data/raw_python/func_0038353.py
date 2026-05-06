def plot_spectrum(self, spectrum_id, table='spectra', column='spectrum', overplot=False, color='b', norm=False):
        """
        Plots a spectrum from the given column and table

        Parameters
        ----------
        spectrum_id: int
            The id from the table of the spectrum to plot.
        overplot: bool
            Overplot the spectrum
        table: str
            The table from which the plot is being made
        column: str
            The column with SPECTRUM data type to plot
        color: str
            The color used for the data
        norm: bool, sequence
            True or (min,max) wavelength range in which to normalize the spectrum

        """
        # TODO: Look into axes number formats. As is it will sometimes not display any numbers for wavelength

        i = self.query("SELECT * FROM {} WHERE id={}".format(table, spectrum_id), fetch='one', fmt='dict')
        if i:
            try:
                spec = scrub(i[column].data, units=False)
                w, f = spec[:2]
                try:
                    e = spec[2]
                except:
                    e = ''

                # Draw the axes and add the metadata
                if not overplot:
                    fig, ax = plt.subplots()
                    plt.rc('text', usetex=False)
                    ax.set_yscale('log', nonposy='clip')
                    plt.figtext(0.15, 0.88, '\n'.join(['{}: {}'.format(k, v) for k, v in i.items() if k != column]), \
                                verticalalignment='top')
                    try:
                        ax.set_xlabel(r'$\lambda$ [{}]'.format(i.get('wavelength_units')))
                        ax.set_ylabel(r'$F_\lambda$ [{}]'.format(i.get('flux_units')))
                    except:
                        pass
                    ax.legend(loc=8, frameon=False)
                else:
                    ax = plt.gca()

                # Normalize the data
                if norm:
                    try:
                        if isinstance(norm, bool): norm = (min(w), max(w))

                        # Normalize to the specified window
                        norm_mask = np.logical_and(w >= norm[0], w <= norm[1])
                        C = 1. / np.trapz(f[norm_mask], x=w[norm_mask])
                        f *= C
                        try:
                            e *= C
                        except:
                            pass

                    except:
                        print('Could not normalize.')

                # Plot the data
                ax.loglog(w, f, c=color, label='spec_id: {}'.format(i['id']))
                X, Y = plt.xlim(), plt.ylim()
                try:
                    ax.fill_between(w, f - e, f + e, color=color, alpha=0.3), ax.set_xlim(X), ax.set_ylim(Y)
                except:
                    print('No uncertainty array for spectrum {}'.format(spectrum_id))
                plt.ion()

            except IOError:
                print("Could not plot spectrum {}".format(spectrum_id))
                plt.close()

        else:
            print("No spectrum {} in the {} table.".format(spectrum_id, table.upper()))
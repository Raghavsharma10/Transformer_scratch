def show_image(self, image_id, table='images', column='image', overplot=False, cmap='hot', log=False):
        """
        Plots a spectrum from the given column and table

        Parameters
        ----------
        image_id: int
            The id from the table of the images to plot.
        overplot: bool
            Overplot the image
        table: str
            The table from which the plot is being made
        column: str
            The column with IMAGE data type to plot
        cmap: str
            The colormap used for the data
        """
        # TODO: Look into axes number formats. As is it will sometimes not display any numbers for wavelength

        i = self.query("SELECT * FROM {} WHERE id={}".format(table, image_id), fetch='one', fmt='dict')
        if i:
            try:
                img = i['image'].data
                
                # Draw the axes and add the metadata
                if not overplot:
                    fig, ax = plt.subplots()
                    plt.rc('text', usetex=False)
                    plt.figtext(0.15, 0.88, '\n'.join(['{}: {}'.format(k, v) for k, v in i.items() if k != column]), \
                                verticalalignment='top')
                    ax.legend(loc=8, frameon=False)
                else:
                    ax = plt.gca()
                    
                # Plot the data
                cmap = plt.get_cmap(cmap)
                cmap.set_under(color='white')
                vmin = 0.0000001
                vmax = np.nanmax(img)
                if log:
                    from matplotlib.colors import LogNorm
                    ax.imshow(img, cmap=cmap, norm=LogNorm(vmin=vmin,vmax=vmax), interpolation='none')
                else:
                    ax.imshow(img, cmap=cmap, interpolation='none', vmin=0.0000001)
                X, Y = plt.xlim(), plt.ylim()
                plt.ion()
                
            except IOError:
                print("Could not plot image {}".format(image_id))
                plt.close()
                
        else:
            print("No image {} in the {} table.".format(image_id, table.upper()))
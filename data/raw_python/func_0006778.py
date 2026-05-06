def figure_populate(outputpath, csv, xlabels, ylabels, analysistype, description, fail=False):
        """
        Create the report image from the summary report created in self.dataframesetup
        :param outputpath: Path in which the outputs are to be created
        :param csv: Name of the report file from which data are to be extracted
        :param xlabels: List of all the labels to use on the x-axis
        :param ylabels: List of all the labels to use on the y-axis
        :param analysistype: String of the analysis type
        :param description: String describing the analysis: set to either template for the empty heatmap created prior
        to analyses or report for normal functionality
        :param fail: Boolean of whether any samples have failed the quality checks - used for determining the palette
        """
        # Create a data frame from the summary report
        df = pd.read_csv(
            os.path.join(outputpath, csv),
            delimiter=',',
            index_col=0)
        # Set the palette appropriately - 'template' uses only grey
        if description == 'template':
            cmap = ['#a0a0a0']
        # 'fail' uses red (fail), grey (not detected), and green (detected/pass)
        elif fail:
            cmap = ['#ff0000', '#a0a0a0', '#00cc00']
        # Otherwise only use grey (not detected) and green (detected/pass)
        else:
            cmap = ['#a0a0a0', '#00cc00']
        # Use seaborn to create a heatmap of the data
        plot = sns.heatmap(df,
                           cbar=False,
                           linewidths=.5,
                           cmap=cmap)
        # Move the x-axis to the top of the plot
        plot.xaxis.set_ticks_position('top')
        # Remove the y-labels
        plot.set_ylabel('')
        # Set the x-tick labels as a slice of the x-labels list (first entry is not required, as it makes the
        # report image look crowded. Rotate the x-tick labels 90 degrees
        plot.set_xticklabels(xlabels[1:], rotation=90)
        # Set the y-tick labels from the supplied list
        plot.set_yticklabels(ylabels, rotation=0)
        # Create the figure
        fig = plot.get_figure()
        # Save the figure in .png format, using the bbox_inches='tight' option to ensure that everything is scaled
        fig.savefig(os.path.join(outputpath, '{at}_{desc}.png'.format(at=analysistype,
                                                                      desc=description)),
                    bbox_inches='tight'
                    )
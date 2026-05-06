def _displayFeatures(self, fig, features, minX, maxX, offsetAdjuster):
        """
        Add the given C{features} to the figure in C{fig}.

        @param fig: A matplotlib figure.
        @param features: A C{FeatureList} instance.
        @param minX: The smallest x coordinate.
        @param maxX: The largest x coordinate.
        @param offsetAdjuster: a function for adjusting feature X axis offsets
            for plotting.
        """
        labels = []
        for index, feature in enumerate(features):
            fig.plot([offsetAdjuster(feature.start),
                      offsetAdjuster(feature.end)],
                     [index * -0.2, index * -0.2], color=feature.color,
                     linewidth=2)
            labels.append(feature.legendLabel())

        # Note that minX and maxX do not need to be adjusted by the offset
        # adjuster. They are the already-adjusted min/max values as
        # computed in computePlotInfo in blast.py
        fig.axis([minX, maxX, (len(features) + 1) * -0.2, 0.2])

        if labels:
            # Put a legend above the figure.
            box = fig.get_position()
            fig.set_position([box.x0, box.y0,
                              box.width, box.height * 0.2])
            fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, 1.4),
                       fancybox=True, shadow=True, ncol=2)
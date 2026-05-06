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
        frame = None
        labels = []
        for feature in features:
            start = offsetAdjuster(feature.start)
            end = offsetAdjuster(feature.end)
            if feature.subfeature:
                subfeatureFrame = start % 3
                if subfeatureFrame == frame:
                    # Move overlapping subfeatures down a little to make them
                    # visible.
                    y = subfeatureFrame - 0.2
                else:
                    y = subfeatureFrame
            else:
                frame = start % 3
                # If we have a polyprotein, shift it up slightly so we can see
                # its components below it.
                product = feature.feature.qualifiers.get('product', [''])[0]
                if product.lower().find('polyprotein') > -1:
                    y = frame + 0.2
                else:
                    y = frame
            fig.plot([start, end], [y, y], color=feature.color, linewidth=2)
            labels.append(feature.legendLabel())

        # Note that minX and maxX do not need to be adjusted by the offset
        # adjuster. They are the already-adjusted min/max values as
        # computed in computePlotInfo in blast.py
        fig.axis([minX, maxX, -0.5, 2.5])
        fig.set_yticks(np.arange(3))
        fig.set_ylabel('Frame')

        if labels:
            # Put a legend above the figure.
            box = fig.get_position()
            fig.set_position([box.x0, box.y0,
                              box.width, box.height * 0.3])
            fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, 2.5),
                       fancybox=True, shadow=True, ncol=2)
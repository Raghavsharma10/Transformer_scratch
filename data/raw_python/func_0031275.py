def plotAAProperties(sequence, propertyNames, showLines=True, showFigure=True):
    """
    Plot amino acid property values for a sequence.

    @param sequence: An C{AARead} (or a subclass) instance.
    @param propertyNames: An iterable of C{str} property names (each of which
        must be a key of a key in the C{dark.aa.PROPERTY_DETAILS} C{dict}).
    @param showLines: If C{True}, lines will be drawn between successive AA
        property values. If not, just the values will be plotted as a scatter
        plot (this greatly reduces visual clutter if the sequence is long and
        AA property values are variable).
    @param showFigure: If C{True}, display the plot. Passing C{False} is useful
        in testing.
    @raise ValueError: If an unknown property is given in C{propertyNames}.
    @return: The return value from calling dark.aa.propertiesForSequence:
        a C{dict} keyed by (lowercase) property name, with values that are
        C{list}s of the corresponding property value according to sequence
        position.
    """
    MISSING_AA_VALUE = -1.1
    propertyValues = propertiesForSequence(sequence, propertyNames,
                                           missingAAValue=MISSING_AA_VALUE)
    if showFigure:
        legend = []
        x = np.arange(0, len(sequence))
        plot = plt.plot if showLines else plt.scatter

        for index, propertyName in enumerate(propertyValues):
            color = TABLEAU20[index]
            plot(x, propertyValues[propertyName], color=color)
            legend.append(patches.Patch(color=color, label=propertyName))

        plt.legend(handles=legend, loc=(0, 1.1))
        plt.xlim(-0.2, len(sequence) - 0.8)
        plt.ylim(min(MISSING_AA_VALUE, -1.1), 1.1)
        plt.xlabel('Sequence index')
        plt.ylabel('Property value')
        plt.title(sequence.id)
        plt.show()

    return propertyValues
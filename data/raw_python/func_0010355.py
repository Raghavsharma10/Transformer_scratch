def plotF0(fromTuple, toTuple, mergeTupleList, fnFullPath):
    '''
    Plots the original data in a graph above the plot of the dtw'ed data
    '''
    _matplotlibCheck()
    
    plt.hold(True)

    fig, (ax0) = plt.subplots(nrows=1)

    # Old data
    plot1 = ax0.plot(fromTuple[0], fromTuple[1], color='red',
                     linewidth=2, label="From")
    plot2 = ax0.plot(toTuple[0], toTuple[1], color='blue',
                     linewidth=2, label="To")
    ax0.set_title("Plot of F0 Morph")
    plt.ylabel('Pitch (hz)')
    plt.xlabel('Time (s)')

    # Merge data
    colorValue = 0
    colorStep = 255.0 / len(mergeTupleList)
    for timeList, valueList in mergeTupleList:
        colorValue += colorStep
        hexValue = "#%02x0000" % int(255 - colorValue)
        if int(colorValue) == 255:
            ax0.plot(timeList, valueList, color=hexValue, linewidth=1,
                     label="Merged line, final iteration")
        else:
            ax0.plot(timeList, valueList, color=hexValue, linewidth=1)

    plt.legend(loc=1, borderaxespad=0.)
#     plt.legend([plot1, plot2, plot3], ["From", "To", "Merged line"])

    plt.savefig(fnFullPath, dpi=300, bbox_inches='tight')
    plt.close(fig)
def howPlotAsk(goodFormat):
    '''plots using inquirer prompts

    Arguments:
        goodFormat {dict} -- module : [results for module]
    '''
    plotAnswer = askPlot()
    if "Save" in plotAnswer['plotQ']:
        exportPlotsPath = pathlib.Path(askSave())
        if "Show" in plotAnswer['plotQ']:
            plotter(exportPlotsPath, True, goodFormat)
        else:
            plotter(exportPlotsPath, False, goodFormat)
    elif "Show" in plotAnswer['plotQ']:
        plotter(None, True, goodFormat)
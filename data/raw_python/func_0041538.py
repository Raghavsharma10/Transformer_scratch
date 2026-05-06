def howPlotArgs(goodFormat):
    '''plots using argparse if can, if not uses howPlotask()

    Arguments:
        goodFormat {dict} -- module : [results for module]
    '''
    if args.exportplots is not None:
        exportPlotsPath = pathlib.Path(args.exportplots)

        if args.showplots:
            plotter(exportPlotsPath, True, goodFormat)
        else:
            plotter(exportPlotsPath, False, goodFormat)
    elif args.showplots:
        plotter(None, True, goodFormat)
    else:
        howPlotAsk(goodFormat)
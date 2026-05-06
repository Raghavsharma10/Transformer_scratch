def process_star(filename, output, *, extension, star_name, period, shift,
                 parameters, period_label, shift_label, **kwargs):
    """Processes a star's lightcurve, prints its coefficients, and saves
    its plotted lightcurve to a file. Returns the result of get_lightcurve.
    """
    if star_name is None:
        basename = path.basename(filename)
        if basename.endswith(extension):
            star_name = basename[:-len(extension)]
        else:
            # file has wrong extension
            return
    if parameters is not None:
        if period is None:
            try:
                period = parameters[period_label][star_name]
            except KeyError:
                pass
            if shift is None:
                try:
                    shift = parameters.loc[shift_label][star_name]
                except KeyError:
                    pass

    result = get_lightcurve_from_file(filename, name=star_name,
                                      period=period, shift=shift,
                                      **kwargs)
    if result is None:
        return
    if output is not None:
        plot_lightcurve(star_name, result['lightcurve'], result['period'],
                        result['phased_data'], output=output, **kwargs)

    return result
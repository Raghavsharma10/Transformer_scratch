def plot_result(x_p, y_p, y_p_e, smoothed_data, smoothed_data_diff, filename=None):
    ''' Fit spline to the profile histogramed data, differentiate, determine MPV and plot.
     Parameters
    ----------
        x_p, y_p : array like
            data points (x,y)
        y_p_e : array like
            error bars in y
    '''
    logging.info('Plot results')
    plt.close()

    p1 = plt.errorbar(x_p * analysis_configuration['vcal_calibration'], y_p, yerr=y_p_e, fmt='o')  # plot data with error bars
    p2, = plt.plot(x_p * analysis_configuration['vcal_calibration'], smoothed_data, '-r')  # plot smoothed data
    factor = np.amax(y_p) / np.amin(smoothed_data_diff) * 1.1
    p3, = plt.plot(x_p * analysis_configuration['vcal_calibration'], factor * smoothed_data_diff, '-', lw=2)  # plot differentiated data
    mpv_index = np.argmax(-analysis_utils.smooth_differentiation(x_p, y_p, weigths=1 / y_p_e, order=3, smoothness=analysis_configuration['smoothness'], derivation=1))
    p4, = plt.plot([x_p[mpv_index] * analysis_configuration['vcal_calibration'], x_p[mpv_index] * analysis_configuration['vcal_calibration']], [0, factor * smoothed_data_diff[mpv_index]], 'k-', lw=2)
    text = 'MPV ' + str(int(x_p[mpv_index] * analysis_configuration['vcal_calibration'])) + ' e'
    plt.text(1.01 * x_p[mpv_index] * analysis_configuration['vcal_calibration'], -10. * smoothed_data_diff[mpv_index], text, ha='left')
    plt.legend([p1, p2, p3, p4], ['data', 'smoothed spline', 'spline differentiation', text], prop={'size': 12}, loc=0)
    plt.title('\'Single hit cluster\'-occupancy for different pixel thresholds')
    plt.xlabel('Pixel threshold [e]')
    plt.ylabel('Single hit cluster occupancy [a.u.]')
    plt.ylim(0, np.amax(y_p) * 1.15)
    if filename is None:
        plt.show()
    else:
        filename.savefig(plt.gcf())
    return smoothed_data_diff
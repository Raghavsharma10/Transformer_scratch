def plot_main(pid, return_fig_ax=False):
    """Main function for creating these plots.

    Reads in plot info dict from json file or dictionary in script.

    Args:
        return_fig_ax (bool, optional): Return figure and axes objects.

    Returns:
        2-element tuple containing
            - **fig** (*obj*): Figure object for customization outside of those in this program.
            - **ax** (*obj*): Axes object for customization outside of those in this program.

    """

    global WORKING_DIRECTORY, SNR_CUT

    if isinstance(pid, PlotInput):
        pid = pid.return_dict()

    WORKING_DIRECTORY = '.'
    if 'WORKING_DIRECTORY' not in pid['general'].keys():
        pid['general']['WORKING_DIRECTORY'] = '.'

    SNR_CUT = 5.0
    if 'SNR_CUT' not in pid['general'].keys():
        pid['general']['SNR_CUT'] = SNR_CUT

    if "switch_backend" in pid['general'].keys():
        plt.switch_backend(pid['general']['switch_backend'])

    running_process = MakePlotProcess(
        **{**pid, **pid['general'], **pid['plot_info'], **pid['figure']})

    running_process.input_data()
    running_process.setup_figure()
    running_process.create_plots()

    # save or show figure
    if 'save_figure' in pid['figure'].keys():
        if pid['figure']['save_figure'] is True:
            running_process.fig.savefig(
                pid['general']['WORKING_DIRECTORY'] + '/' + pid['figure']['output_path'],
                **pid['figure']['savefig_kwargs'])

    if 'show_figure' in pid['figure'].keys():
        if pid['figure']['show_figure'] is True:
            plt.show()

    if return_fig_ax is True:
        return running_process.fig, running_process.ax

    return
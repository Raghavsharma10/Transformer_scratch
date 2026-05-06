def generate_timing_breakdown_plot(timing_stats, scaling_var, title, description, plot_file):
    """
    Description

    Args:
        timing_stats: a dictionary of the form
            {proc_count : {model||bench : { var : { stat : val }}}}
        scaling_var: the variable that accounts for the total runtime
        title: the title of the plot
        description: the description of the plot
        plot_file: the file to write the plot out to
    Returns:
        an image element containing the plot file and metadata
    """
    # noinspection PyProtectedMember
    cmap_data = colormaps._viridis_data
    n_subplots = len(six.viewkeys(timing_stats))
    fig, ax = plt.subplots(1, n_subplots+1, figsize=(3*(n_subplots+2), 5))
    for plot_num, p_count in enumerate(
            sorted(six.iterkeys(timing_stats), key=functions.sort_processor_counts)):

        case_data = timing_stats[p_count]
        all_timers = set(six.iterkeys(case_data['model'])) | set(six.iterkeys(case_data['bench']))
        all_timers = sorted(list(all_timers), reverse=True)
        cmap_stride = int(len(cmap_data)/(len(all_timers)+1))
        colors = {all_timers[i]: cmap_data[i*cmap_stride] for i in range(len(all_timers))}

        sub_ax = plt.subplot(1, n_subplots+1, plot_num+1)
        sub_ax.set_title(p_count)
        sub_ax.set_ylabel('Runtime (s)')
        for case, var_data in case_data.items():
            if case == 'bench':
                bar_num = 2
            else:
                bar_num = 1

            offset = 0
            if var_data != {}:
                for var in sorted(six.iterkeys(var_data), reverse=True):
                    if var != scaling_var:
                        plt.bar(bar_num, var_data[var]['mean'], 0.8, bottom=offset,
                                color=colors[var], label=(var if bar_num == 1 else '_none'))
                        offset += var_data[var]['mean']

                plt.bar(bar_num, var_data[scaling_var]['mean']-offset, 0.8, bottom=offset,
                        color=colors[scaling_var], label=(scaling_var if bar_num == 1 else '_none'))

                sub_ax.set_xticks([1.4, 2.4])
                sub_ax.set_xticklabels(('test', 'bench'))

    plt.legend(loc=6, bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()

    sub_ax = plt.subplot(1, n_subplots+1, n_subplots+1)
    hid_bar = plt.bar(1, 100)
    for group in hid_bar:
            group.set_visible(False)
    sub_ax.set_visible(False)

    if livvkit.publish:
        plt.savefig(os.path.splitext(plot_file)[0]+'.eps', dpi=600)
    plt.savefig(plot_file)
    plt.close()
    return elements.image(title, description, os.path.basename(plot_file))
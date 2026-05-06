def plot_profile(ribo_counts, transcript_name, transcript_length,
                 start_stops, read_lengths=None, read_offsets=None, rna_counts=None,
                 color_scheme='default', html_file='index.html', output_path='output'):
    """Plot read counts (in all 3 frames) and RNA coverage if provided for a
    single transcript.

    """
    colors = get_color_palette(scheme=color_scheme)
    gs = gridspec.GridSpec(3, 1, height_ratios=[6, 1.3, 0.5], hspace=0.35)
    font_axis = {'family': 'sans-serif', 'color': colors['color'], 'weight': 'bold', 'size': 7}

    # riboseq bar plots
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    ax2 = plt.subplot(gs2[0])
    label = 'Ribo-Seq count'

    if read_lengths:
        if len(read_lengths) > 1:
            label = 'Ribo-Seq count ({}-mers)'.format(', '.join('{}'.format(item) for item in read_lengths))
        else:
            label = 'Ribo-Seq count ({}-mer)'.format('{}'.format(read_lengths[0]))

    ax2.set_ylabel(label, fontdict=font_axis, labelpad=10)

    # rna coverage if available
    ax_rna = None
    if rna_counts:
        ax_rna = ax2.twinx()
        ax_rna.set_ylabel('RNA-Seq count', fontdict=font_axis, labelpad=10)
        ax_rna.bar(rna_counts.keys(), rna_counts.values(), facecolor=colors['rna'],
                   edgecolor=colors['rna'], label='RNA')
        ax_rna.set_zorder(1)

    frame_counts = {1: {}, 2: {}, 3: {}}
    for k, v in ribo_counts.iteritems():
        for fr in (1, 2, 3):
            if v[fr] > 0:
                frame_counts[fr][k] = v[fr]
                break

    cnts = []
    [cnts.extend(item.values()) for item in frame_counts.values()]
    y_max = float(max(cnts) * 1.25)
    ax2.set_ylim(0.0, y_max)
    ax2.set_zorder(2)
    ax2.patch.set_facecolor('none')

    for frame in (1, 2, 3):
        color = colors['frames'][frame - 1]
        x_vals = frame_counts[frame].keys()
        ax2.bar(x_vals, frame_counts[frame].values(), color=color, facecolor=color, edgecolor=color)

    # ORF architecture
    gs3 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.1)
    if color_scheme == 'greyorfs':
        axisbg = [colors['grey'] for i in range(3)]
    else:
        axisbg = colors['frames']

    ax4 = plt.subplot(gs3[0], sharex=ax2, axisbg=axisbg[0])
    ax5 = plt.subplot(gs3[1], sharex=ax2, axisbg=axisbg[1])
    ax6 = plt.subplot(gs3[2], sharex=ax2, axisbg=axisbg[2])
    ax6.set_xlabel('Transcript length ({} nt)'.format(transcript_length), fontdict=font_axis, labelpad=6)

    # Legend
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2], hspace=0.1)
    ax7 = plt.subplot(gs4[0], axisbg=colors['background'])
    set_axis_color(ax7, colors['background'])

    ax7.text(0.02, 0.1, "AUG", size=5, ha="center", va="center", color=colors['color'],
             bbox=dict(boxstyle="square", facecolor=colors['start'], edgecolor=colors['color'], linewidth=0.3))
    ax7.text(0.06, 0.1, "STOP", size=5, ha="center", va="center", color='white',
             bbox=dict(boxstyle="square", color=colors['stop']))
    ax7.text(0.13, 0.1, "Frames", size=5, ha='center', va='center', color=colors['color'],
             fontdict={'weight': 'bold'})
    ax7.text(0.17, 0.1, "1", size=5, ha="center", va="center", color='white',
             bbox=dict(boxstyle="square", color=colors['frames'][0]))
    ax7.text(0.19, 0.1, "2", size=5, ha="center", va="center", color='white',
             bbox=dict(boxstyle="square", color=colors['frames'][1]))
    ax7.text(0.21, 0.1, "3", size=5, ha="center", va="center", color='white',
             bbox=dict(boxstyle="square", color=colors['frames'][2]))

    # No ticks or labels for ORF 1, 2 and Legend
    for axis in (ax4, ax5, ax7):
        axis.tick_params(top=False, left=False, right=False, bottom=False, labeltop=False,
                         labelleft=False, labelright=False, labelbottom=False)

    axes = [ax2]
    if ax_rna:
        axes.append(ax_rna)

    fp = FontProperties(size='5')
    for axis in axes:
        set_axis_color(axis, colors['axis'])
        axis.tick_params(colors=colors['ticks'])
        for item in (axis.get_xticklabels() + axis.get_yticklabels()):
            item.set_fontproperties(fp)
            item.set_color(colors['color'])

    for axis, frame in ((ax4, 1), (ax5, 2), (ax6, 3)):
        if color_scheme == 'greyorfs':
            color = colors['grey']
        else:
            color = colors['frames'][frame - 1]
        set_axis_color(axis, color, alpha=0.05)
        axis.patch.set_alpha(0.3)  # opacity of ORF architecture
        for item in (axis.get_xticklabels()):
            item.set_fontproperties(fp)
            item.set_color(colors['color'])
        axis.set_ylim(0, 0.2)
        axis.set_xlim(0, transcript_length)
        starts = [(item, 1) for item in start_stops[frame]['starts']]
        stops = [(item, 1) for item in start_stops[frame]['stops']]
        start_colors = [colors['start'] for item in starts]
        axis.broken_barh(starts, (0.11, 0.2), facecolors=start_colors,
                         edgecolors=start_colors, label='start', zorder=5)
        stop_colors = [colors['stop'] for item in stops]
        axis.broken_barh(stops, (0, 0.2), facecolors=stop_colors,
                         edgecolors=stop_colors, label='stop', zorder=5)
        axis.set_ylabel('{}'.format(frame),
                        fontdict={'family': 'sans-serif', 'color': colors['color'],
                                  'weight': 'normal', 'size': '6'},
                        rotation='horizontal', labelpad=10, verticalalignment='center')
        axis.tick_params(top=False, left=False, right=False, labeltop=False,
                         labelleft=False, labelright=False, direction='out', colors=colors['ticks'])
    plt.title('{}'.format(transcript_name),
              fontdict={'family': 'sans-serif', 'color': colors['color'],
                        'weight': 'bold', 'size': 8, 'y': 20})
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, 'riboplot.svg'), facecolor=colors['background'])
    plt.savefig(os.path.join(output_path, 'riboplot.png'), dpi=600, facecolor=colors['background'])

    with open(os.path.join(CONFIG.PKG_DATA_DIR, 'riboplot.html')) as g, open(os.path.join(output_path, html_file), 'w') as h:
        h.write(g.read().format(transcript_name=transcript_name))

    css_dir = os.path.join(output_path, 'css')
    if not os.path.exists(css_dir):
        os.mkdir(css_dir)

    css_data_dir = os.path.join(CONFIG.PKG_DATA_DIR, 'css')
    for fname in os.listdir(css_data_dir):
        shutil.copy(os.path.join(css_data_dir, fname), os.path.join(output_path, 'css', fname))
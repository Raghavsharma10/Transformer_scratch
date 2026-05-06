def plot_accuracy(data, output_dir_path='.', output_filename='accuracy.png',
                  width=10, height=8):
    """Plot accuracy.
        Args:
            data: Panda dataframe in *the* format.
    """
    output_path = os.path.join(output_dir_path, output_filename)

    max_val_data = get_epoch_max_val_acc(data)
    max_val_label = round(max_val_data['acc'].values[0], 4)

    # max_val_epoch = max_val_data['epoch'].values[0]
    max_epoch_data = data[data['epoch'] == data['epoch'].max()]

    plot = ggplot(data, aes('epoch', 'acc', color='factor(data)')) + \
        geom_line(size=1, show_legend=False) + \
        geom_vline(aes(xintercept='epoch', color='data'),
                   data=max_val_data, alpha=0.5, show_legend=False) + \
        geom_label(aes('epoch', 'acc'), data=max_val_data,
                   label=max_val_label, nudge_y=-0.02, va='top', label_size=0,
                   show_legend=False) + \
        geom_text(aes('epoch', 'acc', label='data'), data=max_epoch_data,
                  nudge_x=2, ha='center', show_legend=False) + \
        geom_point(aes('epoch', 'acc'), data=max_val_data,
                   show_legend=False) + \
        labs(y='Accuracy', x='Epochs') + \
        theme_bw(base_family='Arial', base_size=15) + \
        scale_color_manual(['#ef8a62', '#67a9cf', "#f7f7f7"])

    plot.save(output_path, width=width, height=height)
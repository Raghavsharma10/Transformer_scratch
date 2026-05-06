def plot_bit_for_bit(case, var_name, model_data, bench_data, diff_data):
    """ Create a bit for bit plot """
    plot_title = ""
    plot_name = case + "_" + var_name + ".png"
    plot_path = os.path.join(os.path.join(livvkit.output_dir, "verification", "imgs"))
    functions.mkdir_p(plot_path)
    m_ndim = np.ndim(model_data)
    b_ndim = np.ndim(bench_data)
    if m_ndim != b_ndim:
        return "Dataset dimensions didn't match!"
    if m_ndim == 3:
        model_data = model_data[-1]
        bench_data = bench_data[-1]
        diff_data = diff_data[-1]
        plot_title = "Showing "+var_name+"[-1,:,:]"
    elif m_ndim == 4:
        model_data = model_data[-1][0]
        bench_data = bench_data[-1][0]
        diff_data = diff_data[-1][0]
        plot_title = "Showing "+var_name+"[-1,0,:,:]"
    plt.figure(figsize=(12, 3), dpi=80)
    plt.clf()

    # Calculate min and max to scale the colorbars
    _max = np.amax([np.amax(model_data), np.amax(bench_data)])
    _min = np.amin([np.amin(model_data), np.amin(bench_data)])

    # Plot the model output
    plt.subplot(1, 3, 1)
    plt.xlabel("Model Data")
    plt.ylabel(var_name)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(model_data, vmin=_min, vmax=_max, interpolation='nearest', cmap=colormaps.viridis)
    plt.colorbar()

    # Plot the benchmark data
    plt.subplot(1, 3, 2)
    plt.xlabel("Benchmark Data")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(bench_data, vmin=_min, vmax=_max, interpolation='nearest', cmap=colormaps.viridis)
    plt.colorbar()

    # Plot the difference
    plt.subplot(1, 3, 3)
    plt.xlabel("Difference")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(diff_data, interpolation='nearest', cmap=colormaps.viridis)
    plt.colorbar()

    plt.tight_layout(rect=(0, 0, 0.95, 0.9))
    plt.suptitle(plot_title)

    plot_file = os.path.sep.join([plot_path, plot_name])
    if livvkit.publish:
        plt.savefig(os.path.splitext(plot_file)[0]+'.eps', dpi=600)
    plt.savefig(plot_file)
    plt.close()
    return os.path.join(os.path.relpath(plot_path,
                                        os.path.join(livvkit.output_dir, "verification")),
                        plot_name)
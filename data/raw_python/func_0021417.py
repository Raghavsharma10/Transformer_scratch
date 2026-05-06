def output_barplot(df, figformat, path, title=None, palette=None):
    """Create barplots based on number of reads and total sum of nucleotides sequenced."""
    logging.info("Nanoplotter: Creating barplots for number of reads and total throughput.")
    read_count = Plot(path=path + "NanoComp_number_of_reads." + figformat,
                      title="Comparing number of reads")
    ax = sns.countplot(x="dataset",
                       data=df,
                       palette=palette)
    ax.set(ylabel='Number of reads',
           title=title or read_count.title)
    plt.xticks(rotation=30, ha='center')
    read_count.fig = ax.get_figure()
    read_count.save(format=figformat)
    plt.close("all")

    throughput_bases = Plot(path=path + "NanoComp_total_throughput." + figformat,
                            title="Comparing throughput in gigabases")
    if "aligned_lengths" in df:
        throughput = df.groupby('dataset')['aligned_lengths'].sum()
        ylabel = 'Total gigabase aligned'
    else:
        throughput = df.groupby('dataset')['lengths'].sum()
        ylabel = 'Total gigabase sequenced'
    ax = sns.barplot(x=list(throughput.index),
                     y=throughput / 1e9,
                     palette=palette,
                     order=df["dataset"].unique())
    ax.set(ylabel=ylabel,
           title=title or throughput_bases.title)
    plt.xticks(rotation=30, ha='center')
    throughput_bases.fig = ax.get_figure()
    throughput_bases.save(format=figformat)
    plt.close("all")
    return read_count, throughput_bases
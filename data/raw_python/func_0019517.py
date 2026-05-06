def match_plot(plotdata, outfile):
    """Plot list of motifs with database match and p-value
    "param plotdata: list of (motif, dbmotif, pval)
    """
    fig_h = 2 
    fig_w = 7

    nrows = len(plotdata)
    ncols = 2
    fig = plt.figure(figsize=(fig_w, nrows * fig_h))
    
    for i, (motif, dbmotif, pval) in enumerate(plotdata):
        text = "Motif: %s\nBest match: %s\np-value: %0.2e" % (motif.id, dbmotif.id, pval)
        

        grid = ImageGrid(fig, (nrows, ncols, i * 2 + 1), 
                         nrows_ncols = (2,1),
                         axes_pad=0, 
                         )

        for j in range(2):  
            axes_off(grid[j])

        tmp = NamedTemporaryFile(dir=mytmpdir(), suffix=".png")
        motif.to_img(tmp.name, fmt="PNG", height=6)
        grid[0].imshow(plt.imread(tmp.name), interpolation="none")
        tmp = NamedTemporaryFile(dir=mytmpdir(), suffix=".png")
        dbmotif.to_img(tmp.name, fmt="PNG")
        grid[1].imshow(plt.imread(tmp.name), interpolation="none")

        ax = plt.subplot(nrows, ncols, i * 2 + 2)
        axes_off(ax)

        ax.text(0, 0.5, text,
        horizontalalignment='left',
        verticalalignment='center') 
    
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
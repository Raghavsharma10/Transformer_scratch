def motif_tree_plot(outfile, tree, data, circle=True, vmin=None, vmax=None, dpi=300):
    """
    Plot a "phylogenetic" tree 
    """
    try:
        from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle
    except ImportError:
        print("Please install ete3 to use this functionality")
        sys.exit(1)

    # Define the tree
    t, ts = _get_motif_tree(tree, data, circle, vmin, vmax)
    
    # Save image
    t.render(outfile, tree_style=ts, w=100, dpi=dpi, units="mm");
    
    # Remove the bottom (empty) half of the figure
    if circle:
        img = Image.open(outfile)
        size = img.size[0]
        spacer = 50
        img.crop((0,0,size,size/2 + spacer)).save(outfile)
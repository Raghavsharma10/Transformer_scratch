def get_text_mark(ttree):
    """ makes a simple Text Mark object"""
    
    if ttree._orient in ["right"]:
        angle = 0.
        ypos = ttree.verts[-1*len(ttree.tree):, 1]
        if ttree._kwargs["tip_labels_align"]:
            xpos = [ttree.verts[:, 0].max()] * len(ttree.tree)
            start = xpos
            finish = ttree.verts[-1*len(ttree.tree):, 0]
            align_edges = np.array([(i, i+len(xpos)) for i in range(len(xpos))])
            align_verts = np.array(zip(start, ypos) + zip(finish, ypos))
        else:
            xpos = ttree.verts[-1*len(ttree.tree):, 0]
            
    elif ttree._orient in ['down']:
        angle = -90.
        xpos = ttree.verts[-1*len(ttree.tree):, 0]
        if ttree._kwargs["tip_labels_align"]:
            ypos = [ttree.verts[:, 1].min()] * len(ttree.tree)
            start = ypos
            finish = ttree.verts[-1*len(ttree.tree):, 1]
            align_edges = np.array([(i, i+len(ypos)) for i in range(len(ypos))])
            align_verts = np.array(zip(xpos, start) + zip(xpos, finish))
        else:
            ypos = ttree.verts[-1*len(ttree.tree):, 1]
    
    table = toyplot.data.Table()
    table['x'] = toyplot.require.scalar_vector(xpos)
    table['y'] = toyplot.require.scalar_vector(ypos, table.shape[0])
    table['text'] = toyplot.broadcast.pyobject(ttree.get_tip_labels(), table.shape[0])
    table["angle"] = toyplot.broadcast.scalar(angle, table.shape[0])
    table["opacity"] = toyplot.broadcast.scalar(1.0, table.shape[0])
    table["title"] = toyplot.broadcast.pyobject(None, table.shape[0])
    style = toyplot.style.require(ttree._kwargs["tip_labels_style"],
                                  allowed=toyplot.style.allowed.text)
    default_color = [toyplot.color.black]
    color = toyplot.color.broadcast(
        colors=ttree._kwargs["tip_labels_color"],
        shape=(table.shape[0], 1),
        default=default_color,
        )
    table["fill"] = color[:, 0]
    
    text_mark = toyplot.mark.Text(
        coordinate_axes=['x', 'y'],
        table=table,
        coordinates=['x', 'y'],
        text=["text"],
        angle=["angle"],
        fill=["fill"],
        opacity=["opacity"],
        title=["title"],
        style=style,
        annotation=True,
        filename=None
        )
    return text_mark
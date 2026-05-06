def stitch(folder, filenames, x_size, y_size, output_filename,
                 x_start=0, y_start=0, overlap=10):
    """
    Creates a Fiji Grid/Collection stitching macro. Parameters are the same as
    in the plugin and are described in further detail here:
    http://fiji.sc/Image_Stitching#Grid.2FCollection_Stitching.

    **Default stitch parameters:**

    * Filename defined positions
    * Compute overlap
    * Subpixel accurancy
    * Save computation time (but use more RAM)
    * Fusion method: Linear blending
    * Regression threshold: 0.30
    * Max/avg displacement threshold: 2.50
    * Absolute displacement threshold: 3.50


    Parameters
    ----------
    folder : string
        Path to folder with images or folders with images.
        Example: */path/to/slide--S00/chamber--U01--V02/*
    filenames : string
        Filenames of images.
        Example: *field-X{xx}-Y{yy}/image-X{xx}-Y{yy}.ome.tif*
    x_size : int
        Size of grid, number of images in x direction.
    y_size : int
        Size of grid, number of images in y direction.
    output_filename : string
        Where to store fused image. Should be `.png`.
    x_start : int
        Which x position grid start with.
    y_start : int
        Which y position grid start with.
    overlap : number
        Tile overlap in percent. Fiji will find the optimal overlap, but a
        precise overlap assumption will decrase computation time.

    Returns
    -------
    string
        IJM-macro.
    """

    macro = []
    macro.append('run("Grid/Collection stitching",')
    macro.append('"type=[Filename defined position]')
    macro.append('order=[Defined by filename         ]')
    macro.append('grid_size_x={}'.format(x_size))
    macro.append('grid_size_y={}'.format(y_size))
    macro.append('tile_overlap={}'.format(overlap))
    macro.append('first_file_index_x={}'.format(x_start))
    macro.append('first_file_index_y={}'.format(y_start))
    macro.append('directory=[{}]'.format(folder))
    macro.append('file_names=[{}]'.format(filenames))
    macro.append('output_textfile_name=TileConfiguration.txt')
    macro.append('fusion_method=[Linear Blending]')
    macro.append('regression_threshold=0.20')
    macro.append('max/avg_displacement_threshold=2.50')
    macro.append('absolute_displacement_threshold=3.50')
    macro.append('compute_overlap')
    macro.append('subpixel_accuracy')
    macro.append('computation_parameters=[Save computation time (but use more RAM)]')
    # use display, such that we can specify output filename
    # this is 'Fused and display' for previous stitching version!!
    macro.append('image_output=[Fuse and display]");')
    # save to png
    macro.append('selectWindow("Fused");')
    macro.append('saveAs("PNG", "{}");'.format(output_filename))
    macro.append('close();')

    return ' '.join(macro)
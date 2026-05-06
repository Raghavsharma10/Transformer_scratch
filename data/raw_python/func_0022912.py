def isosurface(data, level):
    """
    Generate isosurface from volumetric data using marching cubes algorithm.
    See Paul Bourke, "Polygonising a Scalar Field"  
    (http://paulbourke.net/geometry/polygonise/)
    
    *data*   3D numpy array of scalar values
    *level*  The level at which to generate an isosurface
    
    Returns an array of vertex coordinates (Nv, 3) and an array of 
    per-face vertex indexes (Nf, 3)    
    """
    # For improvement, see:
    # 
    # Efficient implementation of Marching Cubes' cases with topological 
    # guarantees.
    # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
    # Journal of Graphics Tools 8(2): pp. 1-15 (december 2003)

    (face_shift_tables, edge_shifts, 
     edge_table, n_table_faces) = _get_data_cache()
    
    ## mark everything below the isosurface level
    mask = data < level

    # Because we make use of the strides data attribute below, we have to make 
    # sure that the data is contiguous (which it won't be if the user did 
    # data.transpose() for example). Note that this doesn't copy the data if it 
    # is already contiguous.
    data = np.ascontiguousarray(data)

    ### make eight sub-fields and compute indexes for grid cells
    index = np.zeros([x-1 for x in data.shape], dtype=np.ubyte)
    fields = np.empty((2, 2, 2), dtype=object)
    slices = [slice(0, -1), slice(1, None)]
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                fields[i, j, k] = mask[slices[i], slices[j], slices[k]]
                # this is just to match Bourk's vertex numbering scheme:
                vertIndex = i - 2*j*i + 3*j + 4*k
                index += (fields[i, j, k] * 2**vertIndex).astype(np.ubyte)
    
    ### Generate table of edges that have been cut
    cut_edges = np.zeros([x+1 for x in index.shape]+[3], dtype=np.uint32)
    edges = edge_table[index]
    for i, shift in enumerate(edge_shifts[:12]):        
        slices = [slice(shift[j], cut_edges.shape[j]+(shift[j]-1)) 
                  for j in range(3)]
        cut_edges[slices[0], slices[1], slices[2], shift[3]] += edges & 2**i
    
    # for each cut edge, interpolate to see where exactly the edge is cut and 
    # generate vertex positions
    m = cut_edges > 0
    vertex_inds = np.argwhere(m)  # argwhere is slow!
    vertexes = vertex_inds[:, :3].astype(np.float32).copy()
    dataFlat = data.reshape(data.shape[0]*data.shape[1]*data.shape[2])
    
    ## re-use the cut_edges array as a lookup table for vertex IDs
    cut_edges[vertex_inds[:, 0], 
              vertex_inds[:, 1], 
              vertex_inds[:, 2], 
              vertex_inds[:, 3]] = np.arange(vertex_inds.shape[0])
    
    for i in [0, 1, 2]:
        vim = vertex_inds[:, 3] == i
        vi = vertex_inds[vim, :3]
        vi_flat = (vi * (np.array(data.strides[:3]) // 
                         data.itemsize)[np.newaxis, :]).sum(axis=1)
        v1 = dataFlat[vi_flat]
        v2 = dataFlat[vi_flat + data.strides[i]//data.itemsize]
        vertexes[vim, i] += (level-v1) / (v2-v1)
    
    ### compute the set of vertex indexes for each face. 
    
    ## This works, but runs a bit slower.
    ## all cells with at least one face:
    #cells = np.argwhere((index != 0) & (index != 255))  
    #cellInds = index[cells[:, 0], cells[:, 1], cells[:, 2]]
    #verts = faceTable[cellInds]
    #mask = verts[..., 0, 0] != 9
    ## we now have indexes into cut_edges:
    #verts[...,:3] += cells[:, np.newaxis, np.newaxis,:]
    #verts = verts[mask]
    ## and these are the vertex indexes we want:
    #faces = cut_edges[verts[..., 0], verts[..., 1], verts[..., 2], 
    #                  verts[..., 3]]  
    
    # To allow this to be vectorized efficiently, we count the number of faces 
    # in each grid cell and handle each group of cells with the same number 
    # together.
    
    # determine how many faces to assign to each grid cell
    n_faces = n_table_faces[index]
    tot_faces = n_faces.sum()
    faces = np.empty((tot_faces, 3), dtype=np.uint32)
    ptr = 0
    
    ## this helps speed up an indexing operation later on
    cs = np.array(cut_edges.strides)//cut_edges.itemsize
    cut_edges = cut_edges.flatten()

    ## this, strangely, does not seem to help.
    #ins = np.array(index.strides)/index.itemsize
    #index = index.flatten()

    for i in range(1, 6):
        # expensive:
        # all cells which require i faces  (argwhere is expensive)
        cells = np.argwhere(n_faces == i)  
        if cells.shape[0] == 0:
            continue
        # index values of cells to process for this round:
        cellInds = index[cells[:, 0], cells[:, 1], cells[:, 2]]
        
        # expensive:
        verts = face_shift_tables[i][cellInds]
        # we now have indexes into cut_edges:
        verts[..., :3] += (cells[:, np.newaxis,
                                 np.newaxis, :]).astype(np.uint16)
        verts = verts.reshape((verts.shape[0]*i,)+verts.shape[2:])
        
        # expensive:
        verts = (verts * cs[np.newaxis, np.newaxis, :]).sum(axis=2)
        vert_inds = cut_edges[verts]
        nv = vert_inds.shape[0]
        faces[ptr:ptr+nv] = vert_inds  # .reshape((nv, 3))
        ptr += nv
        
    return vertexes, faces
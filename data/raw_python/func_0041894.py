def convert_coord(coord_from,matrix_file,base_to_aligned=True):
    '''Takes an XYZ array (in DICOM coordinates) and uses the matrix file produced by 3dAllineate to transform it. By default, the 3dAllineate
    matrix transforms from base to aligned space; to get the inverse transform set ``base_to_aligned`` to ``False``'''
    with open(matrix_file) as f:
        try:
            values = [float(y) for y in ' '.join([x for x in f.readlines() if x.strip()[0]!='#']).strip().split()]
        except:
            nl.notify('Error reading values from matrix file %s' % matrix_file, level=nl.level.error)
            return False
    if len(values)!=12:
        nl.notify('Error: found %d values in matrix file %s (expecting 12)' % (len(values),matrix_file), level=nl.level.error)
        return False
    matrix = np.vstack((np.array(values).reshape((3,-1)),[0,0,0,1]))
    if not base_to_aligned:
        matrix = np.linalg.inv(matrix)
    return np.dot(matrix,list(coord_from) + [1])[:3]
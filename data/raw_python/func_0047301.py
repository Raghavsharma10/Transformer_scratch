def get_parcellation(atlas, parcel_param):
    "Placeholder to insert your own function to return parcellation in reference space."
    
    parc_path = os.path.join(atlas, 'parcellation_param{}.mgh'.format(parcel_param))
    parcel = nibabel.freesurfer.io.read_geometry(parc_path)
    
    return parcel
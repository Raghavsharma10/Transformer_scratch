def get_features(subject_id):
    "Placeholder to insert your own function to read subject-wise features."
    
    features_path = os.path.join(my_project,'base_features', subject_id, 'features.txt')
    feature_vector = np.loadtxt(features_path)
    
    return feature_vector
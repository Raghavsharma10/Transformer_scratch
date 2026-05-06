def normalize_features(features):
    """Standardizes features array to fall between 0 and 1"""
    return (features - N.min(features)) / (N.max(features) - N.min(features))
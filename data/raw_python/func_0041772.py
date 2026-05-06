def extract_feature_order_from_model_xml(file_path):
    """
    Takes the path to a FeatureIDE model.xml file and extracts the feature order.
    :param file_path: path to the model file
    :return: list of features as strings
    """

    root = xml.etree.ElementTree.parse(file_path).getroot()
    feature_order_node = root.find('featureOrder')

    features = []
    for child in feature_order_node:
        features.append(child.attrib['name'])

    return features
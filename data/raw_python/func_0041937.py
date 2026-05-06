def get_ordered_feature_list(info_object, feature_list):
    """
    Orders the passed feature list by the given, json-formatted feature
    dependency file using feaquencer's topsort algorithm.
    :param feature_list:
    :param info_object:
    :return:
    """
    feature_dependencies = json.load(open(info_object.feature_order_json))
    feature_selection = [feature for feature in [feature.strip().replace('\n', '') for feature in feature_list]
                         if len(feature) > 0 and not feature.startswith('_') and not feature.startswith('#')]
    return [feature + '\n' for feature in feaquencer.get_total_order(feature_selection, feature_dependencies)]
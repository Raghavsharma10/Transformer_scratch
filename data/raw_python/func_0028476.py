def get_serial_numbers(assetList):
    """
    Helper function: Uses return of get_dev_asset_details function to evaluate to evaluate for multipe serial objects.
    :param assetList: output of get_dev_asset_details function
    :return: the serial_list object of list type which contains one or more dictionaries of the asset details
    """
    serial_list = []
    if type(assetList) == list:
        for i in assetList:
            if len(i['serialNum']) > 0:
                serial_list.append(i)
    return serial_list
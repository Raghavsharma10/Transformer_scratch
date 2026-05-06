def transfer_config_dict(soap_object, data_dict):
    """
    This is a utility function used in the certification modules to transfer
    the data dicts above to SOAP objects. This avoids repetition and allows
    us to store all of our variable configuration here rather than in
    each certification script.
    """
    for key, val in data_dict.items():
        # Transfer each key to the matching attribute ont he SOAP object.
        setattr(soap_object, key, val)
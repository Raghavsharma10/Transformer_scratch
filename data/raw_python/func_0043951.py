def enumerate(vendor_id=0, product_id=0):
    """ Enumerate the HID Devices.

    Returns a generator that yields all of the HID devices attached to the
    system.

    :param vendor_id:   Only return devices which match this vendor id
    :type vendor_id:    int
    :param product_id:  Only return devices which match this product id
    :type product_id:   int
    :return:            Generator that yields informations about attached
                        HID devices
    :rval:              generator(DeviceInfo)

    """
    info = hidapi.hid_enumerate(vendor_id, product_id)
    while info:
        yield DeviceInfo(info)
        info = info.next
    hidapi.hid_free_enumeration(info)
def DeviceFactory(id, lib=None):
    """Create the correct device instance based on device type and return it.

    :return: a :class:`Device` or :class:`DeviceGroup` instance.
    """
    lib = lib or Library()
    if lib.tdGetDeviceType(id) == const.TELLSTICK_TYPE_GROUP:
        return DeviceGroup(id, lib=lib)
    return Device(id, lib=lib)
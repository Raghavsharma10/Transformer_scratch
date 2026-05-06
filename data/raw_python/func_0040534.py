def createDbusProxyObject(bus_name, object_path, bus=None):
    '''
    Create dbus proxy object
    '''
    bus = bus or dbus.SessionBus.get_session()
    return bus.get_object(bus_name, object_path)
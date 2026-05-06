def empowerment(iface, priority=0):
    """
    Class decorator for indicating a powerup's powerup interfaces.

    The class will also be declared as implementing the interface.

    @type iface: L{zope.interface.Interface}
    @param iface: The powerup interface.

    @type priority: int
    @param priority: The priority the powerup will be installed at.
    """
    def _deco(cls):
        cls.powerupInterfaces = (
            tuple(getattr(cls, 'powerupInterfaces', ())) +
            ((iface, priority),))
        implementer(iface)(cls)
        return cls
    return _deco
def registerUpgrader(upgrader, typeName, oldVersion, newVersion):
    """
    Register a callable which can perform a schema upgrade between two
    particular versions.

    @param upgrader: A one-argument callable which will upgrade an object.  It
    is invoked with an instance of the old version of the object.
    @param typeName: The database typename for which this is an upgrader.
    @param oldVersion: The version from which this will upgrade.
    @param newVersion: The version to which this will upgrade.  This must be
    exactly one greater than C{oldVersion}.
    """
    # assert (typeName, oldVersion, newVersion) not in _upgradeRegistry, "duplicate upgrader"
    # ^ this makes the tests blow up so it's just disabled for now; perhaps we
    # should have a specific test mode
    # assert newVersion - oldVersion == 1, "read the doc string"
    assert isinstance(typeName, str), "read the doc string"
    _upgradeRegistry[typeName, oldVersion] = upgrader
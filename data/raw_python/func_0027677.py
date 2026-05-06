def processor(forType):
    """
    Create an Axiom Item type which is suitable to use as a batch processor for
    the given Axiom Item type.

    Processors created this way depend on a L{iaxiom.IScheduler} powerup on the
    on which store they are installed.

    @type forType: L{item.MetaItem}
    @param forType: The Axiom Item type for which to create a batch processor
    type.

    @rtype: L{item.MetaItem}

    @return: An Axiom Item type suitable for use as a batch processor.  If such
    a type previously existed, it will be returned.  Otherwise, a new type is
    created.
    """
    MILLI = 1000
    try:
        processor = _processors[forType]
    except KeyError:
        def __init__(self, *a, **kw):
            item.Item.__init__(self, *a, **kw)
            self.store.powerUp(self, iaxiom.IBatchProcessor)

        attrs = {
            '__name__': 'Batch_' + forType.__name__,

            '__module__': forType.__module__,

            '__init__': __init__,

            '__repr__': lambda self: '<Batch of %s #%d>' % (reflect.qual(self.workUnitType), self.storeID),

            'schemaVersion': 2,

            'workUnitType': forType,

            'scheduled': attributes.timestamp(doc="""
            The next time at which this processor is scheduled to run.
            """, default=None),

            # MAGIC NUMBERS AREN'T THEY WONDERFUL?
            'busyInterval': attributes.integer(doc="", default=MILLI / 10),
            }
        _processors[forType] = processor = item.MetaItem(
            attrs['__name__'],
            (item.Item, _BatchProcessorMixin),
            attrs)

        registerUpgrader(
            upgradeProcessor1to2,
            _processors[forType].typeName,
            1, 2)

    return processor
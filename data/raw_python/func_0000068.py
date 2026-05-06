def service(flavour):
    r"""
    Mark a class as implementing a Service

    Each Service class must have a ``run`` method, which does not take any arguments.
    This method is :py:meth:`~.ServiceRunner.adopt`\ ed after the daemon starts, unless

    * the Service has been garbage collected, or
    * the ServiceUnit has been :py:meth:`~.ServiceUnit.cancel`\ ed.

    For each service instance, its :py:class:`~.ServiceUnit` is available at ``service_instance.__service_unit__``.
    """
    def service_unit_decorator(raw_cls):
        __new__ = raw_cls.__new__

        def __new_service__(cls, *args, **kwargs):
            if __new__ is object.__new__:
                self = __new__(cls)
            else:
                self = __new__(cls, *args, **kwargs)
            service_unit = ServiceUnit(self, flavour)
            self.__service_unit__ = service_unit
            return self

        raw_cls.__new__ = __new_service__
        if raw_cls.run.__doc__ is None:
            raw_cls.run.__doc__ = "Service entry point"
        return raw_cls
    return service_unit_decorator
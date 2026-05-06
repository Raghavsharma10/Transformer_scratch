def connect(self, detailed_signal, handler, *args):
        """connect(detailed_signal: str, handler: function, *args) -> handler_id: int

        The connect() method adds a function or method (handler) to the end
        of the list of signal handlers for the named detailed_signal but
        before the default class signal handler. An optional set of
        parameters may be specified after the handler parameter. These will
        all be passed to the signal handler when invoked.

        For example if a function handler was connected to a signal using::

            handler_id = object.connect("signal_name", handler, arg1, arg2, arg3)

        The handler should be defined as::

            def handler(object, arg1, arg2, arg3):

        A method handler connected to a signal using::

            handler_id = object.connect("signal_name", self.handler, arg1, arg2)

        requires an additional argument when defined::

            def handler(self, object, arg1, arg2)

        A TypeError exception is raised if detailed_signal identifies a
        signal name that is not associated with the object.
        """

        return self.__connect(0, detailed_signal, handler, *args)
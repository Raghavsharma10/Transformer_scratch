def prepare_notification(self, *, subscribers=None, instance=None,
                             loop=None, notify_external=True):
        """Sets up a and configures an `~.utils.Executor`:class: instance."""
        # merge callbacks added to the class level with those added to the
        # instance, giving the formers precedence while preserving overall
        # order
        self_subscribers = self.subscribers.copy()
        # add in callbacks declared in the main class body and marked with
        # @handler
        if (instance is not None and self.name and
            isinstance(instance.__class__, SignalAndHandlerInitMeta)):
            class_handlers = type(instance)._get_class_handlers(
                self.name, instance)
            for ch in class_handlers:
                # eventual methods are ephemeral and normally the following
                # condition would always be True for methods but the dict used
                # has logic to take that into account
                if ch not in self_subscribers:
                    self_subscribers.append(ch)
        # add in the other instance level callbacks added at runtime
        if subscribers is not None:
            for el in subscribers:
                # eventual methods are ephemeral and normally the following
                # condition would always be True for methods but the dict used
                # has logic to take that into account
                if el not in self_subscribers:
                    self_subscribers.append(el)
        loop = loop or self.loop
        # maybe do a round of external publishing
        if notify_external and self.external_signaller is not None:
            self_subscribers.append(partial(self.ext_publish, instance, loop))
        if self._fnotify is None:
            fnotify = None
        else:
            if instance is None:
                fnotify = self._fnotify
            else:
                fnotify = types.MethodType(self._fnotify, instance)
        validator = self._fvalidation
        if validator is not None and instance is not None:
            validator = types.MethodType(validator, instance)
        return Executor(self_subscribers, owner=self,
                        concurrent=SignalOptions.EXEC_CONCURRENT in self.flags,
                        loop=loop, exec_wrapper=fnotify,
                        fvalidation=validator)
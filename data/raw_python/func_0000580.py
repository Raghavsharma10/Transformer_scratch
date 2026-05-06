def register_payload(self, *payloads, flavour: ModuleType):
        """Queue one or more payload for execution after its runner is started"""
        for payload in payloads:
            self._logger.debug('registering payload %s (%s)', NameRepr(payload), NameRepr(flavour))
            self.runners[flavour].register_payload(payload)
def __collect_interfaces_return(interfaces):
        """Collect new style (44.1+) return values to old-style kv-list"""
        acc = []
        for (interfaceName, interfaceData) in interfaces.items():
            signalValues = interfaceData.get("signals", {})
            for (signalName, signalValue) in signalValues.items():
                pinName = "{0}.{1}".format(interfaceName, signalName)
                acc.append({'id': pinName, 'value': signalValue})
        return acc
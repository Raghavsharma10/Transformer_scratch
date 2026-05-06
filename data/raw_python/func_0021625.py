def isDriver(self):
        """
        Determines if the current L{PE} instance is a driver (.sys) file.
        
        @rtype: bool
        @return: C{True} if the current L{PE} instance is a driver. Otherwise, returns C{False}.
        """
        modules = []
        imports = self.ntHeaders.optionalHeader.dataDirectory[consts.IMPORT_DIRECTORY].info
        for module in imports:
            modules.append(module.metaData.moduleName.value.lower())
        
        if set(["ntoskrnl.exe", "hal.dll", "ndis.sys", "bootvid.dll", "kdcom.dll"]).intersection(modules):
            return True
        return False
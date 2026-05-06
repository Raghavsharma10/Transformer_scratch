def enable_capture_state(self, state, writeToHw=False):
        """
        Enable/Disable capture on resource group
        """
        if state:
            activePorts = self.rePortInList.findall(self.activePortList)
            self.activeCapturePortList = "{{" + activePorts[0] + "}}"
        else:
            self.activeCapturePortList = "{{""}}"
        if (writeToHw):
            self.ix_command('write')
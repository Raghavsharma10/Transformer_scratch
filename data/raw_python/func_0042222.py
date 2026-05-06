def read(self):
        "Read and interpret data from the daemon."
        status = gpscommon.read(self)
        if status <= 0:
            return status
        if self.response.startswith("{") and self.response.endswith("}\r\n"):
            self.unpack(self.response)
            self.__oldstyle_shim()
            self.newstyle = True
            self.valid |= PACKET_SET
        elif self.response.startswith("GPSD"):
            self.__oldstyle_unpack(self.response)
            self.valid |= PACKET_SET
        return 0
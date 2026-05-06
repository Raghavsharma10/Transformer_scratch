def detectS60OssBrowser(self):
        """Return detection of Symbian S60 Browser

        Detects if the current browser is the Symbian S60 Open Source Browser.
        """
        #First, test for WebKit, then make sure it's either Symbian or S60.
        return self.detectWebkit() \
            and (UAgentInfo.deviceSymbian in self.__userAgent
                or UAgentInfo.deviceS60 in self.__userAgent)
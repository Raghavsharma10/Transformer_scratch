def detectMobileLong(self):
        """Return detection of any mobile device using the more thorough method

        The longer and more thorough way to detect for a mobile device.
        Will probably detect most feature phones,
        smartphone-class devices, Internet Tablets,
        Internet-enabled game consoles, etc.
        This ought to catch a lot of the more obscure and older devices, also --
        but no promises on thoroughness!
        """

        if self.detectMobileQuick() \
           or self.detectGameConsole():
            return True

        if self.detectDangerHiptop() \
           or self.detectMaemoTablet() \
           or self.detectSonyMylo() \
           or self.detectArchos():
            return True

        if UAgentInfo.devicePda in self.__userAgent \
           and UAgentInfo.disUpdate not in self.__userAgent:
            return True

        #detect older phones from certain manufacturers and operators.
        return UAgentInfo.uplink in self.__userAgent \
            or UAgentInfo.engineOpenWeb in self.__userAgent \
            or UAgentInfo.manuSamsung1 in self.__userAgent \
            or UAgentInfo.manuSonyEricsson in self.__userAgent \
            or UAgentInfo.manuericsson in self.__userAgent \
            or UAgentInfo.svcDocomo in self.__userAgent \
            or UAgentInfo.svcKddi in self.__userAgent \
            or UAgentInfo.svcVodafone in self.__userAgent
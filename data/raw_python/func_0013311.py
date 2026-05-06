def addLocations(self, effect, protPos, cdnaPos):
        """
        Adds locations to a GA4GH transcript effect object
        by parsing HGVS annotation fields in concert with
        and supplied position values.
        :param effect: protocol.TranscriptEffect
        :param protPos: String representing protein position from VCF
        :param cdnaPos: String representing coding DNA location
        :return: effect protocol.TranscriptEffect
        """
        self.addCDSLocation(effect, cdnaPos)
        self.addCDNALocation(effect, cdnaPos)
        self.addProteinLocation(effect, protPos)
        return effect
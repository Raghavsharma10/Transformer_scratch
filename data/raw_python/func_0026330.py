def _read_header(self):
        """ Process the header information (data model / grid spec) """

        self._header_pos = self.fp.tell()

        line = self.fp.readline('20sffii')
        modelname, res0, res1, halfpolar, center180 = line
        self._attributes.update({
            "modelname": str(modelname, 'utf-8').strip(),
            "halfpolar": halfpolar,
            "center180": center180,
            "res": (res0, res1)
        })
        self.__setattr__('modelname', modelname)
        self.__setattr__('res', (res0, res1))
        self.__setattr__('halfpolar', halfpolar)
        self.__setattr__('center180', center180)

        # Re-wind the file
        self.fp.seek(self._header_pos)
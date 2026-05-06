def print_archive(self,format=True):
        """ Prints out archived WCS keywords."""
        if len(list(self.orig_wcs.keys())) > 0:
            block  = 'Original WCS keywords for ' + self.rootname+ '\n'
            block += '    backed up on '+repr(self.orig_wcs['WCSCDATE'])+'\n'
            if not format:
                for key in self.wcstrans.keys():
                    block += key.upper() + " = " + repr(self.get_archivekw(key)) + '\n'
                block = 'PA_V3: '+repr(self.pa_obs)+'\n'

            else:
                block += 'CD_11  CD_12: '+repr(self.get_archivekw('CD1_1'))+'  '+repr(self.get_archivekw('CD1_2')) +'\n'
                block += 'CD_21  CD_22: '+repr(self.get_archivekw('CD2_1'))+'  '+repr(self.get_archivekw('CD2_2')) +'\n'
                block += 'CRVAL       : '+repr(self.get_archivekw('CRVAL1'))+'  '+repr(self.get_archivekw('CRVAL2')) + '\n'
                block += 'CRPIX       : '+repr(self.get_archivekw('CRPIX1'))+'  '+repr(self.get_archivekw('CRPIX2')) + '\n'
                block += 'NAXIS       : '+repr(int(self.get_archivekw('NAXIS1')))+'  '+repr(int(self.get_archivekw('NAXIS2'))) + '\n'
                block += 'Plate Scale : '+repr(self.get_archivekw('pixel scale'))+'\n'
                block += 'ORIENTAT    : '+repr(self.get_archivekw('ORIENTAT'))+'\n'

            print(block)
def publish(self, **kwargs):
        '''
        Correct the light curve with the CBVs, generate a
        cover page for the DVS figure,
        and produce a FITS file for publication.

        '''

        try:

            # HACK: Force these params for publication
            self.cbv_win = 999
            self.cbv_order = 3
            self.cbv_num = 1

            # Get the CBVs
            self._mission.GetTargetCBVs(self)

            # Plot the final corrected light curve
            cbv = CBV()
            self.plot_info(cbv)
            self.plot_cbv(cbv.body(), self.fcor, 'Corrected')
            self.plot_cbv(cbv.body(), self.flux, 'De-trended', show_cbv=True)
            self.plot_cbv(cbv.body(), self.fraw, 'Raw')

            # Save the CBV pdf
            pdf = PdfPages(os.path.join(self.dir, 'cbv.pdf'))
            pdf.savefig(cbv.fig)
            pl.close(cbv.fig)
            d = pdf.infodict()
            d['Title'] = 'EVEREST: %s de-trending of %s %d' % (
                self.name, self._mission.IDSTRING, self.ID)
            d['Author'] = 'Rodrigo Luger'
            pdf.close()

            # Now merge the two PDFs
            assert os.path.exists(os.path.join(
                self.dir, self.name + '.pdf')), \
                "Unable to locate %s.pdf." % self.name
            output = PdfFileWriter()
            pdfOne = PdfFileReader(os.path.join(self.dir, 'cbv.pdf'))
            pdfTwo = PdfFileReader(os.path.join(self.dir, self.name + '.pdf'))
            # Add the CBV page
            output.addPage(pdfOne.getPage(0))
            # Add the original DVS page
            output.addPage(pdfTwo.getPage(pdfTwo.numPages - 1))
            # Write the final PDF
            outputStream = open(os.path.join(self.dir, self._mission.DVSFile(
                self.ID, self.season, self.cadence)), "wb")
            output.write(outputStream)
            outputStream.close()
            os.remove(os.path.join(self.dir, 'cbv.pdf'))

            # Make the FITS file
            MakeFITS(self)

        except:

            self.exception_handler(self.debug)
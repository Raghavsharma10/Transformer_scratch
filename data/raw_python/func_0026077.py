def createReferenceWCS(self,refname,overwrite=yes):
        """ Write out the values of the WCS keywords to the NEW
            specified image 'fitsname'.

        """
        hdu = self.createWcsHDU()
        # If refname already exists, delete it to make way for new file
        if os.path.exists(refname):
            if overwrite==yes:
                # Remove previous version and re-create with new header
                os.remove(refname)
                hdu.writeto(refname)
            else:
                # Append header to existing file
                wcs_append = True
                oldhdu = fits.open(refname, mode='append')
                for e in oldhdu:
                    if 'extname' in e.header and e.header['extname'] == 'WCS':
                        wcs_append = False
                if wcs_append == True:
                    oldhdu.append(hdu)
                oldhdu.close()
                del oldhdu
        else:
            # No previous file, so generate new one from scratch
            hdu.writeto(refname)

        # Clean up
        del hdu
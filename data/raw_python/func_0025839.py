def write(self, output=None):
        """
        Write association table to a file.

        """
        if not output:
            outfile = self['output']+'_asn.fits'
            output = self['output']
        else:
            outfile = output

        # Delete the file if it exists.
        if os.path.exists(outfile):
            warningmsg =  "\n#########################################\n"
            warningmsg += "#                                       #\n"
            warningmsg += "# WARNING:                              #\n"
            warningmsg += "#  The existing association table,      #\n"
            warningmsg += "           " + str(outfile) + '\n'
            warningmsg += "#  is being replaced.                   #\n"
            warningmsg += "#                                       #\n"
            warningmsg += "#########################################\n\n"
        fasn = fits.HDUList()

        # Compute maximum length of MEMNAME for table column definition
        _maxlen = 0
        for _fname in self['order']:
            if len(_fname) > _maxlen: _maxlen = len(_fname)
        # Enforce a mimimum size of 24
        if _maxlen < 24: _maxlen = 24
        namelen_str = str(_maxlen+2)+'A'
        self.buildPrimary(fasn, output=output)

        mname = self['order'][:]
        mname.append(output)
        mtype = ['EXP-DTH' for l in self['order']]
        mtype.append('PROD-DTH')
        mprsn = [True for l in self['order']]
        mprsn.append(False)
        xoff = [self['members'][l]['xoff'] for l in self['order']]
        xoff.append(0.0)
        yoff = [self['members'][l]['yoff'] for l in self['order']]
        yoff.append(0.0)
        xsh = [self['members'][l]['xshift'] for l in self['order']]
        xsh.append(0.0)
        ysh = [self['members'][l]['yshift'] for l in self['order']]
        ysh.append(0.0)
        rot = [self['members'][l]['rot'] for l in self['order']]
        rot.append(0.0)
        scl = [self['members'][l]['scale'] for l in self['order']]
        scl.append(1.0)

        memname = fits.Column(name='MEMNAME',format=namelen_str,array=N.char.array(mname))
        memtype = fits.Column(name='MEMTYPE',format='14A',array=N.char.array(mtype))
        memprsn = fits.Column(name='MEMPRSNT', format='L', array=N.array(mprsn).astype(N.uint8))
        xoffset = fits.Column(name='XOFFSET', format='E', array=N.array(xoff))
        yoffset = fits.Column(name='YOFFSET', format='E', array=N.array(yoff))
        xdelta = fits.Column(name='XDELTA', format='E', array=N.array(xsh))
        ydelta = fits.Column(name='YDELTA', format='E', array=N.array(ysh))
        rotation = fits.Column(name='ROTATION', format='E', array=N.array(rot))
        scale = fits.Column(name='SCALE', format='E', array=N.array(scl))
        cols = fits.ColDefs([memname,memtype,memprsn,xoffset,yoffset,xdelta,ydelta,rotation,scale])
        hdu = fits.BinTableHDU.from_columns(cols)
        fasn.append(hdu)
        if ASTROPY_VER_GE13:
            fasn.writeto(outfile, overwrite=True)
        else:
            fasn.writeto(outfile, clobber=True)
        fasn.close()
        mem0 = self['order'][0]
        refimg = self['members'][mem0]['refimage']
        if refimg is not None:
            whdu = wcsutil.WCSObject(refimg)
            whdu.createReferenceWCS(outfile,overwrite=False)
            ftab = fits.open(outfile)
            ftab['primary'].header['refimage'] = outfile+"[wcs]"
            ftab.close()
        del whdu
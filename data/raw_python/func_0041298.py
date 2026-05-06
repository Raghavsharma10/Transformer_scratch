def DICOMfile_read(self, *args, **kwargs):
        """
        Read a DICOM file and perform some initial
        parsing of tags.

        NB!
        For thread safety, class member variables
        should not be assigned since other threads
        might override/change these variables in mid-
        flight!

        """
        b_status        = False
        l_tags          = []
        l_tagsToUse     = []
        d_tagsInString  = {}
        str_file        = ""

        d_DICOM           = {
            'dcm':              None,
            'd_dcm':            {},
            'strRaw':           '',
            'l_tagRaw':         [],
            'd_json':           {},
            'd_dicom':          {},
            'd_dicomSimple':    {}
        }

        for k, v in kwargs.items():
            if k == 'file':             str_file    = v
            if k == 'l_tagsToUse':      l_tags      = v

        if len(args):
            l_file          = args[0]
            str_file        = l_file[0]

        str_localFile   = os.path.basename(str_file)
        str_path        = os.path.dirname(str_file)
        # self.dp.qprint("%s: In input base directory:      %s" % (threading.currentThread().getName(), self.str_inputDir))
        # self.dp.qprint("%s: Reading DICOM file in path:   %s" % (threading.currentThread().getName(),str_path))
        # self.dp.qprint("%s: Analysing tags on DICOM file: %s" % (threading.currentThread().getName(),str_localFile))      
        # self.dp.qprint("%s: Loading:                      %s" % (threading.currentThread().getName(),str_file))

        try:
            # self.dcm    = dicom.read_file(str_file)
            d_DICOM['dcm']  = dicom.read_file(str_file)
            b_status    = True
        except:
            self.dp.qprint('In directory: %s' % os.getcwd(),    comms = 'error')
            self.dp.qprint('Failed to read %s' % str_file,      comms = 'error')
            b_status    = False
        d_DICOM['d_dcm']    = dict(d_DICOM['dcm'])
        d_DICOM['strRaw']   = str(d_DICOM['dcm'])
        d_DICOM['l_tagRaw'] = d_DICOM['dcm'].dir()

        if len(l_tags):
            l_tagsToUse     = l_tags
        else:
            l_tagsToUse     = d_DICOM['l_tagRaw']

        if 'PixelData' in l_tagsToUse:
            l_tagsToUse.remove('PixelData')

        for key in l_tagsToUse:
            d_DICOM['d_dicom'][key]       = d_DICOM['dcm'].data_element(key)
            try:
                d_DICOM['d_dicomSimple'][key] = getattr(d_DICOM['dcm'], key)
            except:
                d_DICOM['d_dicomSimple'][key] = "no attribute"
            d_DICOM['d_json'][key]        = str(d_DICOM['d_dicomSimple'][key])

        # pudb.set_trace()
        d_tagsInString  = self.tagsInString_process(d_DICOM, self.str_outputFileStem)
        str_outputFile  = d_tagsInString['str_result']

        return {
            'status':           b_status,
            'inputPath':        str_path,
            'inputFilename':    str_localFile,
            'outputFileStem':   str_outputFile,
            'd_DICOM':          d_DICOM,
            'l_tagsToUse':      l_tagsToUse
        }
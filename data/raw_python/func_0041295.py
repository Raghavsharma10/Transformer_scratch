def declare_selfvars(self):
        """
        A block to declare self variables
        """

        #
        # Object desc block
        #
        self.str_desc                   = ''
        self.__name__                   = "pfdicom"
        self.str_version                = '1.6.0'

        # Directory and filenames
        self.str_workingDir             = ''
        self.str_inputDir               = ''
        self.str_inputFile              = ''
        self.str_extension              = ''
        self.str_outputFileStem         = ''
        self.str_ouptutDir              = ''
        self.str_outputLeafDir          = ''
        self.maxDepth                   = -1

        # pftree dictionary
        self.pf_tree                    = None
        self.numThreads                 = 1

        self.str_stdout                 = ''
        self.str_stderr                 = ''
        self.exitCode                   = 0

        self.b_json                     = False
        self.b_followLinks              = False

        # The actual data volume and slice
        # are numpy ndarrays
        self.dcm                        = None
        self.d_dcm                      = {}     # dict convert of raw dcm
        self.strRaw                     = ""
        self.l_tagRaw                   = []

        # Simpler dictionary representations of DICOM tags
        # NB -- the pixel data is not read into the dictionary
        # by default
        self.d_dicom                   = {}     # values directly from dcm ojbect
        self.d_dicomSimple             = {}     # formatted dict convert

        # Convenience vars
        self.tic_start                  = None

        self.dp                         = None
        self.log                        = None
        self.tic_start                  = 0.0
        self.pp                         = pprint.PrettyPrinter(indent=4)
        self.verbosityLevel             = 1
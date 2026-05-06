def _prepare_outputs(self, data_out, outputs):
        """ Open a ROOT file with option 'RECREATE' to create a new file (the file will
        be overwritten if it already exists), and using the ZLIB compression algorithm
        (with compression level 1) for better compatibility with older ROOT versions
        (see https://root.cern.ch/doc/v614/release-notes.html#important-notice ).

        :param data_out:
        :param outputs:
        :return:
        """
        compress = ROOTModule.ROOT.CompressionSettings(ROOTModule.ROOT.kZLIB, 1)
        if isinstance(data_out, (str, unicode)):
            self.file_emulation = True
            outputs.append(ROOTModule.TFile.Open(data_out, 'RECREATE', '', compress))
        # multiple tables - require directory
        elif isinstance(data_out, ROOTModule.TFile):
            outputs.append(data_out)
        else:  # assume it's a file like object
            self.file_emulation = True
            filename = os.path.join(tempfile.mkdtemp(),'tmp.root')
            outputs.append(ROOTModule.TFile.Open(filename, 'RECREATE', '', compress))
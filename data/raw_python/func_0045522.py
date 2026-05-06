def read_header(self):
        """Hardware configuration A505
        Head configuration A504
        User configuration A500"""
        with open(self.filepath, 'rb') as instrumentDataFile:
            while True:
                sync = instrumentDataFile.read(1)
                if not sync:
                    break
                elif sync == '\xa5':
                    id = instrumentDataFile.read(1)
                    if id == '\x05':
                        instrumentDataFile.seek(instrumentDataFile.tell() - 2)
                        hardwareConfiguration = nortek.structures.Header(
                                instrumentDataFile.read(48)) # always
                        headConfiguration = nortek.structures.Header(
                                instrumentDataFile.read(224)) # always
                        userConfiguration = nortek.structures.Header(
                                instrumentDataFile.read(512)) # always
                        self.endOfConfiguration = instrumentDataFile.tell()
                        if (hardwareConfiguration.checksum and 
                            headConfiguration.checksum and 
                            userConfiguration.checksum):
                            self[ 'hardwareConfiguration' ] = hardwareConfiguration
                            self[ 'headConfiguration' ] = headConfiguration
                            self[ 'userConfiguration' ] = userConfiguration
                            self[ 'type' ] = self[ 'hardwareConfiguration' ].interpretBinaryData()
                            self[ 'headConfiguration' ].interpretBinaryData(self[ 'type' ])
                            self[ 'userConfiguration' ].interpretBinaryData(self[ 'type' ])
                            break
                        else:
                            pdb.set_trace()
                            # there were problems, try to figure out what
                            print("""Checksum failure in the header. Checksum values are 
								hardware: {}
								head: {}
								user: {}
								Data file position is {}""".format(
									self[ 'hardwareConfiguration' ].checksum,
									self[ 'headConfiguration' ].checksum,
									self[ 'userConfiguration' ].checksum,
									instrumentDataFile.tell()))
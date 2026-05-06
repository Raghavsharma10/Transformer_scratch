def reporter(self):
        """
        Creates a report of the results
        """
        logging.info('Creating {} report'.format(self.analysistype))
        # Create the path in which the reports are stored
        make_path(self.reportpath)
        header = 'Strain,Serotype\n'
        data = ''
        with open(os.path.join(self.reportpath, '{}.csv'.format(self.analysistype)), 'w') as report:
            for sample in self.runmetadata.samples:
                if sample.general.bestassemblyfile != 'NA':
                    data += sample.name + ','
                    if sample[self.analysistype].results:
                        # Set the O-type as either the appropriate attribute, or O-untypable
                        if ';'.join(sample.serosippr.o_set) == '-':
                            otype = 'O-untypeable'
                        else:
                            otype = '{oset} ({opid})'.format(oset=';'.join(sample.serosippr.o_set),
                                                             opid=sample.serosippr.best_o_pid)
                        # Same as above, but for the H-type
                        if ';'.join(sample.serosippr.h_set) == '-':
                            htype = 'H-untypeable'

                        else:
                            htype = '{hset} ({hpid})'.format(hset=';'.join(sample.serosippr.h_set),
                                                             hpid=sample.serosippr.best_h_pid)
                        serotype = '{otype}:{htype}'.format(otype=otype,
                                                            htype=htype)
                        # Populate the data string
                        data += serotype if serotype != 'O-untypeable:H-untypeable' else 'ND'
                        data += '\n'
                    else:
                        data += '\n'
            report.write(header)
            report.write(data)
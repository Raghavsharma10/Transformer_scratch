def gdcs_fai(sample, analysistype='GDCS'):
        """
        GDCS analyses need to use the .fai file supplied in the targets folder rather than the one created following
        reverse baiting
        :param sample: sample object
        :param analysistype: current analysis being performed
        """
        try:
            # Find the .fai file in the target path
            sample[analysistype].faifile = glob(os.path.join(sample[analysistype].targetpath, '*.fai'))[0]
        except IndexError:
            target_file = glob(os.path.join(sample[analysistype].targetpath, '*.fasta'))[0]
            samindex = SamtoolsFaidxCommandline(reference=target_file)
            map(StringIO, samindex(cwd=sample[analysistype].targetpath))
            sample[analysistype].faifile = glob(os.path.join(sample[analysistype].targetpath, '*.fai'))[0]
        # Get the fai file into a dictionary to be used in parsing results
        try:
            with open(sample[analysistype].faifile, 'r') as faifile:
                for line in faifile:
                    data = line.split('\t')
                    try:
                        sample[analysistype].faidict[data[0]] = int(data[1])
                    except KeyError:
                        sample[analysistype].faidict = dict()
                        sample[analysistype].faidict[data[0]] = int(data[1])
        except FileNotFoundError:
            pass
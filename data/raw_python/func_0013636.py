def readValuesBigWigToWig(self, reference, start, end):
        """
        Read a bigwig file and return a protocol object with values
        within the query range.

        This method uses the bigWigToWig command line tool from UCSC
        GoldenPath. The tool is used to return values within a query region.
        The output is in wiggle format, which is processed by the WiggleReader
        class.

        There could be memory issues if the returned results are large.

        The input reference can be a security problem (script injection).
        Ideally, it should be checked against a list of known chromosomes.
        Start and end should not be problems since they are integers.
        """
        if not self.checkReference(reference):
            raise exceptions.ReferenceNameNotFoundException(reference)
        if start < 0:
            raise exceptions.ReferenceRangeErrorException(
                reference, start, end)
            # TODO: CHECK IF QUERY IS BEYOND END

        cmd = ["bigWigToWig", self._sourceFile, "stdout", "-chrom="+reference,
               "-start="+str(start), "-end="+str(end)]
        wiggleReader = WiggleReader(reference, start, end)
        try:
            # run command and grab output simultaneously
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break
                wiggleReader.readWiggleLine(line.strip())
        except ValueError:
            raise
        except:
            raise Exception("bigWigToWig failed to run")

        return wiggleReader.getData()
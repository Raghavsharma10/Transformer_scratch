def strains(self):
        """
        Create a dictionary of SEQID: OLNID from the supplied
        """
        with open(os.path.join(self.path, 'strains.csv')) as strains:
            next(strains)
            for line in strains:
                oln, seqid = line.split(',')
                self.straindict[oln] = seqid.rstrip()
                self.strainset.add(oln)
                logging.debug(oln)
                if self.debug:
                    break
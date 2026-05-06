def update_tracking_terms(self):
        """
        Terms must be one-per-line.
        Blank lines will be skipped.
        """
        import codecs
        with codecs.open(self.filename,"r", encoding='utf8') as input:
            # read all the lines
            lines = input.readlines()

            # build a set of terms
            new_terms = set()
            for line in lines:
                line = line.strip()
                if len(line):
                    new_terms.add(line)

            return set(new_terms)
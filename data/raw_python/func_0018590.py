def label_list_parser(self, url):
        """
        Extracts comma separate tag=value pairs from a string
        Assumes all characters other than / and , are valid
        """
        labels = re.findall('([^/,]+=[^/,]+)', url)
        slabels = set(labels)
        if '' in slabels:
            slabels.remove('')
        return slabels
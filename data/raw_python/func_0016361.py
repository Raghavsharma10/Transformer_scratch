def files(self):
        """
        Return the list of available files in the coverage report.
        """
        # maybe replace with a trie at some point? see has_file FIXME
        already_seen = set()
        filenames = []

        for el in self.xml.xpath("//class"):
            filename = el.attrib['filename']
            if filename in already_seen:
                continue
            already_seen.add(filename)
            filenames.append(filename)

        return filenames
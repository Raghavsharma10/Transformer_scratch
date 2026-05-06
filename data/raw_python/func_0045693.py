def write(self):
        """Write csv file of resolved names and txt file of unresolved names.
        """
        csv_file = os.path.join(self.outdir, 'search_results.csv')
        txt_file = os.path.join(self.outdir, 'unresolved.txt')
        headers = self.key_terms
        unresolved = []
        with open(csv_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for key in list(self._store.keys()):
                results = self._store[key]
                if len(results) == 0:
                    unresolved.append(key)
                else:
                    row = [key]
                    for key_term in headers[1:]:
                        element = results[0][key_term]
                        # GNR returns UTF-8, csv requires ascii
                        #
                        # *** Note ***
                        # According to all docs for csv versions >= 2.6, csv
                        # can handle either UTF-8 or ascii, just not Unicode.
                        # In py3, the following two lines result in csv printing
                        # the element with a bitstring. If GNR is actually
                        # returning UTF-8, it seems easiest to just drop these

                        # if 'encode' in dir(element):
                        #     element = element.encode('ascii')
                        row.append(element)
                    writer.writerow(row)
        if len(unresolved) > 0:
            with open(txt_file, 'w') as file:
                for name in unresolved:
                    file.write("{0}\n".format(name))
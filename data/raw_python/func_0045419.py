def search(self, terms, prelim=True):
        """Search terms against GNR. If prelim = False, search other datasources \
for alternative names (i.e. synonyms) with which to search main datasource.\
Return JSON object."""
        # TODO: There are now lots of additional data sources, make additional
        # searching optional (11/01/2017)
        if prelim:  # preliminary search
            res = self._resolve(terms, self.Id)
            self._write(res)
            return res
        else:  # search other DSs for alt names, search DS with these
            # quick fix: https://github.com/DomBennett/TaxonNamesResolver/issues/5
            # seems to be due to limit on number of ids in single request
            # switiching to a for loop for each data source
            # appending all results into single res
            res = []
            for ds_id in self.otherIds:
                tmp = self._resolve(terms, [ds_id])
                res.append(tmp[0])
            self._write(res)
            alt_terms = self._parseNames(res)
            if len(alt_terms) == 0:
                return False
            else:
                # search the main source again with alt_terms
                # replace names in json
                terms = [each[1] for each in alt_terms]  # unzip
                res = self._resolve(terms, self.Id)
                self._write(res)
                alt_res = self._replaceSupStrNames(res, alt_terms)
                return alt_res
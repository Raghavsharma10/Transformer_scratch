def lookup(self, criteria, table, columns=''):
        """
        Returns a table of records from *table* the same length as *criteria*
        with the best match for each element.
        
        Parameters
        ----------
        criteria: sequence 
            The search criteria
        table: str
            The table to search
        columns: sequence
            The column name in the sources table to search
        
        Returns
        -------
        results: sequence
            A sequence the same length as objlist with source_ids that correspond 
            to successful matches and blanks where no matches could be made
        """
        results, colmasks = [], []
        
        # Iterate through the list, trying to match objects
        for n,criterion in enumerate(criteria):
            records = self.search(criterion, table, columns=columns, fetch=True)
                
            # If multiple matches, take the first but notify the user of the other matches
            if len(records)>1:
                print("'{}' matched to {} other record{}.".format(criterion, len(records)-1, \
                      's' if len(records)-1>1 else ''))
            
            # If no matches, make an empty row
            if len(records)==0:
                records.add_row(np.asarray(np.zeros(len(records.colnames))).T)
                colmasks.append([True]*len(records.colnames))
            else:
                colmasks.append([False]*len(records.colnames))
            
            # Grab the first row
            results.append(records[0])
        
        # Add all the rows to the results table
        table = at.Table(rows=results, names=results[0].colnames, masked=True)
        
        # Mask the rows with no matches
        for col,msk in zip(records.colnames,np.asarray(colmasks).T): 
            table[col].mask = msk
        
        return table
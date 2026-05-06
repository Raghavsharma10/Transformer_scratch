def data(self, as_cells=False):
        """ Reads the worksheet and returns an indexed dictionary of the
        row objects.
        
        For example:

        >>>print sheet.data()

        {'Miss Piggy': {'Color': 'Pink', 'Performer': 'Frank Oz'}, 'Kermit': {'Color': 'Green', 'Performer': 'Jim Henson'}} 
        
        """
        sheet_data = {}
        self.max_row = max(self.header_row_ix, self.formula_ref_row_ix)
        all_cells = self._cell_feed(row=self.max_row+1,
                                    further_rows=True,
                                    col=self.header.first_column,
                                    max_col=self.header.last_column,
                                    return_empty=True)

        for wks_row in self._yield_rows(all_cells):
            if wks_row.row_num not in sheet_data and not wks_row.is_empty():
                sheet_data[wks_row.row_num] = wks_row

        all_rows = sheet_data.keys()
        if all_rows:
            self.max_row = max(all_rows)

        # Now index by key_tuple
        indexed_sheet_data = {}
        for row, wks_row in sheet_data.iteritems():
            # Make the key tuple
            if len(self.key_column_headers) == 0:
                # Are there any default key column headers?
                if "Key" in wks_row:
                    logger.info("Assumed key column's header is 'Key'")
                    self.key_column_headers = ['Key']
                elif "Key-1" in wks_row:
                    self.key_column_headers = [h for h in wks_row.keys() 
                        if h.startswith("Key-") and h.split("-")[1].isdigit()]
                    logger.info("Assumed key column headers were: %s",
                                self.key_column_headers)
                else:
                    raise Exception("Unable to read spreadsheet. Specify"
                        "key_column_headers when initializing Sheet object.")

            key_list = []
            for key_hdr in self.key_column_headers:
                key_val = wks_row.db.get(key_hdr,"")
                if key_val.startswith("'"):
                    key_val = key_val[1:]
                key_list.append(key_val)
            key_tuple = tuple(key_list)
            if all(k == "" for k in key_tuple):
                continue

            if as_cells:
                indexed_sheet_data[key_tuple] = wks_row
            else:
                if len(key_tuple) == 1:
                    key_tuple = key_tuple[0]
                indexed_sheet_data[key_tuple] = wks_row.db

        return indexed_sheet_data
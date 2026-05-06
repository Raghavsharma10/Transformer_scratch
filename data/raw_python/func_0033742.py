def print_tables(xmldoc, output, output_format, tableList = [], columnList = [],
    round_floats = True, decimal_places = 2, format_links = True,
    title = None, print_table_names = True, unique_rows = False,
    row_span_columns = [], rspan_break_columns = []):
    """
    Method to print tables in an xml file in other formats.
    Input is an xmldoc, output is a file object containing the
    tables.

    @xmldoc: document to convert
    @output: file object to write output to; if None, will write to stdout
    @output_format: format to convert to
    @tableList: only convert the listed tables. Default is
     to convert all the tables found in the xmldoc. Tables
     not converted will not be included in the returned file
     object.
    @columnList: only print the columns listed, in the order given. 
     This applies to all tables (if a table doesn't have a listed column, it's just
     skipped). To specify a column in a specific table, use table_name:column_name.
     Default is to print all columns.
    @round_floats: If turned on, will smart_round floats to specifed
     number of places.
    @format_links: If turned on, will convert any html hyperlinks to specified
     output_format.
    @decimal_places: If round_floats turned on, will smart_round to this
     number of decimal places.
    @title: Add a title to this set of tables.
    @unique_rows: If two consecutive rows are exactly the same, will condense into
     one row.
    @print_table_names: If set to True, will print the name of each table
     in the caption section.
    @row_span_columns: For the columns listed, will
     concatenate consecutive cells with the same values
     into one cell that spans those rows. Default is to span no rows.
    @rspan_break_column: Columns listed will prevent all cells
     from rowspanning across two rows in which values in the
     columns are diffrent. Default is to have no break columns.
    """
    # get the tables to convert
    if tableList == []:
        tableList = [tb.getAttribute("Name") for tb in xmldoc.childNodes[0].getElementsByTagName(u'Table')]

    # set the output
    if output is None:
        output = sys.stdout

    # get table bits
    ttx, xtt, tx, xt, capx, xcap, rx, xr, cx, xc, rspx, xrsp, hlx, hxl, xhl = set_output_format( output_format )

    # set the title if desired
    if title is not None:
        print >> output, "%s%s%s" %(ttx,str(title),xtt)
    # cycle over the tables in the xmldoc
    for table_name in tableList:
        this_table = table.get_table(xmldoc, table_name)
        if columnList == []:
            col_names = [ col.getAttribute("Name").split(":")[-1]
                for col in this_table.getElementsByTagName(u'Column') ]
        else:
            requested_columns = [col.split(':')[-1] for col in columnList if not (':' in col and col.split(':')[0] != table_name) ]
            requested_columns = sorted(set(requested_columns), key=requested_columns.index)
            actual_columns = [actual_column.getAttribute("Name").split(":")[-1]
                for actual_column in this_table.getElementsByTagName(u'Column') ]
            col_names = [col for col in requested_columns if col in actual_columns]
        # get the relevant row_span/break column indices
        rspan_indices = [ n for n,col in enumerate(col_names) if col in row_span_columns or ':'.join([table_name,col]) in row_span_columns ]
        break_indices = [ n for n,col in enumerate(col_names) if col in rspan_break_columns or ':'.join([table_name,col]) in rspan_break_columns ] 

        # start the table and print table name
        print >> output, tx
        if print_table_names:
            print >> output, "%s%s%s" %(capx, table_name, xcap)
        print >> output, "%s%s%s%s%s" %(rx, cx, (xc+cx).join(format_header_cell(val) for val in col_names), xc, xr)

        # format the data in the table
        out_table = []
        last_row = ''
        for row in this_table:
            out_row = [ str(format_cell( get_row_data(row, col_name),
                round_floats = round_floats, decimal_places = decimal_places,
                format_links = format_links,  hlx = hlx, hxl = hxl, xhl = xhl ))
                for col_name in col_names ]
            if unique_rows and out_row == last_row:
                continue
            out_table.append(out_row)
            last_row = out_row

        rspan_count = {}
        for mm, row in enumerate(out_table[::-1]):
            this_row_idx = len(out_table) - (mm+1)
            next_row_idx = this_row_idx - 1
            # cheack if it's ok to do row-span
            rspan_ok = rspan_indices != [] and this_row_idx != 0
            if rspan_ok:
                for jj in break_indices:
                    rspan_ok = out_table[this_row_idx][jj] == out_table[next_row_idx][jj]
                    if not rspan_ok: break
            # cycle over columns in the row setting row span values
            for nn, val in enumerate(row):
                # check if this cell should be spanned;
                # if so, delete it, update rspan_count and go on to next cell
                if rspan_ok and nn in rspan_indices:
                    if val == out_table[next_row_idx][nn]:
                        out_table[this_row_idx][nn] = ''
                        if (this_row_idx, nn) in rspan_count:
                            rspan_count[(next_row_idx,nn)] = rspan_count[(this_row_idx,nn)] + 1
                            del rspan_count[(this_row_idx,nn)]
                        else:
                            rspan_count[(next_row_idx,nn)] = 2 
                    elif (this_row_idx, nn) in rspan_count:
                        out_table[this_row_idx][nn] = ''.join([rspx, str(rspan_count[(this_row_idx,nn)]), xrsp, str(val), xc])
                    else:
                        out_table[this_row_idx][nn] = ''.join([cx, str(val), xc])
                    continue
                # format cell appropriately
                if (this_row_idx, nn) in rspan_count:
                    out_table[this_row_idx][nn] = ''.join([rspx, str(rspan_count[(this_row_idx,nn)]), xrsp, str(val), xc])
                else:
                    out_table[this_row_idx][nn] = ''.join([cx, str(val), xc])

        # print the table to output
        for row in out_table:
            print >> output, "%s%s%s" % (rx, ''.join(row), xr)

        # close the table and go on to the next
        print >> output, xt
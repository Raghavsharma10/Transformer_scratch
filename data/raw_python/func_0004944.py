def listB1(fsns, xlsname, dirs, whattolist = None, headerformat = 'org_%05d.header'):
    """ getsamplenames revisited, XLS output.
    
    Inputs:
        fsns: FSN sequence
        xlsname: XLS file name to output listing
        dirs: either a single directory (string) or a list of directories, a la readheader()
        whattolist: format specifier for listing. Should be a list of tuples. Each tuple
            corresponds to a column in the worksheet, in sequence. The first element of
            each tuple is the column title, eg. 'Distance' or 'Calibrated energy (eV)'.
            The second element is either the corresponding field in the header dictionary
            ('Dist' or 'EnergyCalibrated'), or a tuple of them, eg. ('FSN', 'Title', 'Energy').
            If the column-descriptor tuple does not have a third element, the string
            representation of each field (str(param[i][fieldname])) will be written
            in the corresponding cell. If a third element is present, it is treated as a 
            format string, and the values of the fields are substituted.
        headerformat: C-style format string of header file names (e.g. org_%05d.header)
        
    Outputs:
        an XLS workbook is saved.
    
    Notes:
        if whattolist is not specified exactly (ie. is None), then the output
            is similar to getsamplenames().
        module xlwt is needed in order for this function to work. If it cannot
            be imported, the other functions may work, only this function will
            raise a NotImplementedError.
    """
    if whattolist is None:
        whattolist = [('FSN', 'FSN'), ('Time', 'MeasTime'), ('Energy', 'Energy'),
                    ('Distance', 'Dist'), ('Position', 'PosSample'),
                    ('Transmission', 'Transm'), ('Temperature', 'Temperature'),
                    ('Title', 'Title'), ('Date', ('Day', 'Month', 'Year', 'Hour', 'Minutes'), '%02d.%02d.%04d %02d:%02d')]
    wb = xlwt.Workbook(encoding = 'utf8')
    ws = wb.add_sheet('Measurements')
    for i in range(len(whattolist)):
        ws.write(0, i, whattolist[i][0])
    i = 1
    for fsn in fsns:
        try:
            hed = readB1header(findfileindirs(headerformat % fsn, dirs))
        except IOError:
            continue
        # for each param structure create a line in the table
        for j in range(len(whattolist)):
            # for each parameter to be listed, create a column
            if np.isscalar(whattolist[j][1]):
                # if the parameter is a scalar, make it a list
                fields = tuple([whattolist[j][1]])
            else:
                fields = whattolist[j][1]
            if len(whattolist[j]) == 2:
                if len(fields) >= 2:
                    strtowrite = ''.join([str(hed[f]) for f in fields])
                else:
                    strtowrite = hed[fields[0]]
            elif len(whattolist[j]) >= 3:
                strtowrite = whattolist[j][2] % tuple([hed[f] for f in fields])
            else:
                assert False
            ws.write(i, j, strtowrite)
        i += 1
    wb.save(xlsname)
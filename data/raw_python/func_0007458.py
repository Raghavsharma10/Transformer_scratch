def _load_data(data_file, data_type):
    """Load data from CSV, JSON, Excel, ..., formats."""
    raw_data = data_file.read()
    if data_type is None:
        data_type = data_file.name.split('.')[-1]
    # Data list to process
    data = []
    # JSON type
    if data_type == 'json':
        data = json.loads(raw_data)
        return data
    # CSV type
    elif data_type == 'csv':
        csv_data = StringIO(raw_data)
        reader = csv.DictReader(csv_data, delimiter=',')
        for line in reader:
            data.append(line)
        return data
    elif data_type in ['xlsx', 'xlsm', 'xltx', 'xltm']:
        excel_data = StringIO(raw_data)
        wb = openpyxl.load_workbook(excel_data)
        ws = wb.active
        # First headers
        headers = []
        for row in ws.iter_rows(max_row=1):
            for cell in row:
                tmp = '_'.join(cell.value.split(" ")).lower()
                headers.append(tmp)
        # Simulate DictReader
        for row in ws.iter_rows(row_offset=1):
            values = []
            for cell in row:
                values.append(cell.value)
            tmp = dict(itertools.izip(headers, values))
            if len(values) == len(headers) and not row_empty(values):
                data.append(tmp)
        return data
    # PO type
    elif data_type == 'po':
        po = polib.pofile(raw_data)
        for entry in po.untranslated_entries():
            data.append(entry.__dict__)
        return data
    # PROPERTIES type (used in Java and Firefox extensions)
    elif data_type == 'properties':
        lines = raw_data.split('\n')
        for l in lines:
            if l:
                var_id, string = l.split('=')
                tmp = dict(var_id=var_id, string=string)
                data.append(tmp)
        return data
    else:
        return data
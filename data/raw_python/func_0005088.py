def se_iban_load_map(filename: str) -> list:
    """
    Loads Swedish monetary institution codes in CSV format.
    :param filename: CSV file name of the BIC definitions.
    Columns: Institution Name, Range Begin-Range End (inclusive), Account digits count
    :return: List of (bank name, clearing code begin, clearing code end, account digits)
    """
    out = []
    name_repl = {
        'BNP Paribas Fortis SA/NV, Bankfilial Sverige': 'BNP Paribas Fortis SA/NV',
        'Citibank International Plc, Sweden Branch': 'Citibank International Plc',
        'Santander Consumer Bank AS (deltar endast i Dataclearingen)': 'Santander Consumer Bank AS',
        'Nordax Bank AB (deltar endast i Dataclearingen)': 'Nordax Bank AB',
        'Swedbank och fristående Sparbanker, t ex Leksands Sparbank och Roslagsbanken.': 'Swedbank',
        'Ålandsbanken Abp (Finland),svensk filial': 'Ålandsbanken Abp',
        'SBAB deltar endast i Dataclearingen': 'SBAB',
    }
    with open(filename) as fp:
        for row in csv.reader(fp):
            if len(row) == 3:
                name, series, acc_digits = row
                # pprint([name, series, acc_digits])

                # clean up name
                name = re.sub(r'\n.*', '', name)
                if name in name_repl:
                    name = name_repl[name]

                # clean up series
                ml_acc_digits = acc_digits.split('\n')
                for i, ser in enumerate(series.split('\n')):
                    begin, end = None, None
                    res = re.match(r'^(\d+)-(\d+).*$', ser)
                    if res:
                        begin, end = res.group(1), res.group(2)
                    if begin is None:
                        res = re.match(r'^(\d{4}).*$', ser)
                        if res:
                            begin = res.group(1)
                            end = begin

                    if begin and end:
                        digits = None
                        try:
                            digits = int(acc_digits)
                        except ValueError:
                            pass
                        if digits is None:
                            try:
                                digits = int(ml_acc_digits[i])
                            except ValueError:
                                digits = '?'
                            except IndexError:
                                digits = '?'

                        out.append([name, begin, end, digits])
                        # print('OK!')
    return out
def df2arff(df, dataset_name, pods_data):
    """Write an arff file from a data set loaded in from pods"""
    def java_simple_date(date_format):
        date_format = date_format.replace('%Y', 'yyyy').replace('%m', 'MM').replace('%d', 'dd').replace('%H', 'HH')
        return date_format.replace('%h', 'hh').replace('%M', 'mm').replace('%S', 'ss').replace('%f', 'SSSSSS')
    
    def tidy_field(atr):
        return str(atr).replace(' / ', '/').replace(' ', '_')
    types = {'STRING': [str], 'INTEGER': [int, np.int64, np.uint8], 'REAL': [np.float64]}
    d = {}
    d['attributes'] = []
    for atr in df.columns:
        if isinstance(atr, str):
            if len(atr)>8 and atr[:9] == 'discrete(':
                import json
                elements = json.loads(atr[9:-1])
                d['attributes'].append((tidy_field(elements[1]),
                                         list(elements[0].keys())))
                mask = {}
                c = pd.Series(index=df.index)
                for key, val in elements[0].items():
                    mask = df[atr]==val
                    c[mask] = key
                df[atr] = c
                continue
            if len(atr)>7 and atr[:8] == 'integer(':
                name = atr[8:-1]
                d['attributes'].append((tidy_field(name), 'INTEGER'))
                df[atr] = df[atr].astype(int)
                continue
            if len(atr)>7 and atr[:8]=='datenum(':
                from matplotlib.dates import num2date
                elements = atr[8:-1].split(',')
                d['attributes'].append((elements[0] + '_datenum_' + java_simple_date(elements[1]), 'STRING'))
                df[atr] = num2date(df[atr].values) #
                df[atr] = df[atr].dt.strftime(elements[1])
                continue
            if len(atr)>9 and atr[:10]=='timestamp(':
                def timestamp2date(values):
                    import datetime
                    """Convert timestamp into a date object"""
                    new = []
                    for value in values:
                        new.append(np.datetime64(datetime.datetime.fromtimestamp(value)))
                    return np.asarray(new)
                elements = atr[10:-1].split(',')
                d['attributes'].append((elements[0] + '_datenum_' + java_simple_date(elements[1]), 'STRING'))
                df[atr] = timestamp2date(df[atr].values) #
                df[atr] = df[atr].dt.strftime(elements[1])
                continue
            if len(atr)>10 and atr[:11]=='datetime64(':
                elements = atr[11:-1].split(',')
                d['attributes'].append((elements[0] + '_datenum_' + java_simple_date(elements[1]), 'STRING'))
                df[atr] = df[atr].dt.strftime(elements[1])
                continue
            if len(atr)>11 and atr[:12]=='decimalyear(':
                def decyear2date(values):
                    """Convert decimal year into a date object"""
                    new = []
                    for i, decyear in enumerate(values):
                        year = int(np.floor(decyear))
                        dec = decyear-year
                        end = np.datetime64(str(year+1)+'-01-01')
                        start = np.datetime64(str(year)+'-01-01')
                        diff=end-start
                        days = dec*(diff/np.timedelta64(1, 'D'))
                        # round to nearest day
                        add = np.timedelta64(int(np.round(days)), 'D')
                        new.append(start+add)
                    return np.asarray(new)
                elements = atr[12:-1].split(',')
                d['attributes'].append((elements[0] + '_datenum_' + java_simple_date(elements[1]), 'STRING'))
                df[atr] = decyear2date(df[atr].values) #
                df[atr] = df[atr].dt.strftime(elements[1])
                continue

        field = tidy_field(atr)
        el = df[atr][0]
        type_assigned=False
        for t in types:
            if isinstance(el, tuple(types[t])):
                d['attributes'].append((field, t))
                type_assigned=True
                break
        if not type_assigned:
            import json
            d['attributes'].append((field+'_json', 'STRING'))
            df[atr] = df[atr].apply(json.dumps)

    d['data'] = []
    for ind, row in df.iterrows():
        d['data'].append(list(row))

    import textwrap as tw
    width = 78
    d['description'] = dataset_name + "\n\n"
    if 'info' in pods_data and pods_data['info']:
        d['description'] += "\n".join(tw.wrap(pods_data['info'], width)) + "\n\n"
    if 'details' in pods_data and pods_data['details']:
        d['description'] += "\n".join(tw.wrap(pods_data['details'], width))
    if 'citation' in pods_data and pods_data['citation']:
        d['description'] += "\n\n" + "Citation" "\n\n" + "\n".join(tw.wrap(pods_data['citation'], width))

    d['relation'] = dataset_name
    import arff
    string = arff.dumps(d)
    import re
    string = re.sub(r'\@ATTRIBUTE "?(.*)_datenum_(.*)"? STRING',
                    r'@ATTRIBUTE "\1" DATE [\2]',
                    string)
    f = open(dataset_name + '.arff', 'w')
    f.write(string)
    f.close()
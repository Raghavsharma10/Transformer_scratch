def google_trends(query_terms=['big data', 'machine learning', 'data science'], data_set='google_trends', refresh_data=False):
    """
    Data downloaded from Google trends for given query terms. Warning,
    if you use this function multiple times in a row you get blocked
    due to terms of service violations.

    The function will cache the result of any query in an attempt to
    avoid this. If you wish to refresh an old query set refresh_data
    to True. The function is inspired by this notebook:

    http://nbviewer.ipython.org/github/sahuguet/notebooks/blob/master/GoogleTrends%20meet%20Notebook.ipynb

    """

    query_terms.sort()
    import pandas as pd

    # Create directory name for data
    dir_path = os.path.join(data_path,'google_trends')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    dir_name = '-'.join(query_terms)
    dir_name = dir_name.replace(' ', '_')
    dir_path = os.path.join(dir_path,dir_name)
    file = 'data.csv'
    file_name = os.path.join(dir_path,file)
    if not os.path.exists(file_name) or refresh_data:
        print("Accessing Google trends to acquire the data. Note that repeated accesses will result in a block due to a google terms of service violation. Failure at this point may be due to such blocks.")
        # quote the query terms.
        quoted_terms = []
        for term in query_terms:
            quoted_terms.append(quote(term))
        print("Query terms: ", ', '.join(query_terms))

        print("Fetching query:")
        query = 'http://www.google.com/trends/fetchComponent?q=%s&cid=TIMESERIES_GRAPH_0&export=3' % ",".join(quoted_terms)

        data = urlopen(query).read().decode('utf8')
        print("Done.")
        # In the notebook they did some data cleaning: remove Javascript header+footer, and translate new Date(....,..,..) into YYYY-MM-DD.
        header = """// Data table response\ngoogle.visualization.Query.setResponse("""
        data = data[len(header):-2]
        data = re.sub('new Date\((\d+),(\d+),(\d+)\)', (lambda m: '"%s-%02d-%02d"' % (m.group(1).strip(), 1+int(m.group(2)), int(m.group(3)))), data)
        timeseries = json.loads(data)
        columns = [k['label'] for k in timeseries['table']['cols']]
        rows = list(map(lambda x: [k['v'] for k in x['c']], timeseries['table']['rows']))
        df = pd.DataFrame(rows, columns=columns)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        df.to_csv(file_name)
    else:
        print("Reading cached data for google trends. To refresh the cache set 'refresh_data=True' when calling this function.")
        print("Query terms: ", ', '.join(query_terms))

        df = pd.read_csv(file_name, parse_dates=[0])

    columns = df.columns
    terms = len(query_terms)
    import datetime
    from matplotlib.dates import date2num
    X = np.asarray([(date2num(datetime.datetime.strptime(df.ix[row]['Date'], '%Y-%m-%d')), i) for i in range(terms) for row in df.index])
    Y = np.asarray([[df.ix[row][query_terms[i]]] for i in range(terms) for row in df.index ])
    output_info = columns[1:]
    cats = {}
    for i in range(terms):
        cats[query_terms[i]] = i
    return data_details_return({'data frame' : df, 'X': X, 'Y': Y, 'query_terms': query_terms, 'info': "Data downloaded from google trends with query terms: " + ', '.join(query_terms) + '.', 'covariates' : [datenum('date'), discrete(cats, 'query_terms')], 'response' : ['normalized interest']}, data_set)
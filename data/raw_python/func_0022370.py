def football_data(season='1617', data_set='football_data'):
    """Football data from English games since 1993. This downloads data from football-data.co.uk for the given season. """
    league_dict = {'E0':0, 'E1':1, 'E2': 2, 'E3': 3, 'EC':4}
    def league2num(string):
        if isinstance(string, bytes):
            string = string.decode('utf-8')
        return league_dict[string]

    def football2num(string):
        if isinstance(string, bytes):
            string = string.decode('utf-8')
        if string in football_dict:
            return football_dict[string]
        else:
            football_dict[string] = len(football_dict)+1
            return len(football_dict)+1

    def datestr2num(s):
        import datetime
        from matplotlib.dates import date2num
        return date2num(datetime.datetime.strptime(s.decode('utf-8'),'%d/%m/%y'))
    data_set_season = data_set + '_' + season
    data_resources[data_set_season] = copy.deepcopy(data_resources[data_set])
    data_resources[data_set_season]['urls'][0]+=season + '/'
    start_year = int(season[0:2])
    end_year = int(season[2:4])
    files = ['E0.csv', 'E1.csv', 'E2.csv', 'E3.csv']
    if start_year>4 and start_year < 93:
        files += ['EC.csv']
    data_resources[data_set_season]['files'] = [files]
    if not data_available(data_set_season):
        download_data(data_set_season)
    start = True
    for file in reversed(files):
        filename = os.path.join(data_path, data_set_season, file)
        # rewrite files removing blank rows.
        writename = os.path.join(data_path, data_set_season, 'temp.csv')
        input = open(filename, encoding='ISO-8859-1')
        output = open(writename, 'w')
        writer = csv.writer(output)
        for row in csv.reader(input):
            if any(field.strip() for field in row):
                writer.writerow(row)
        input.close()
        output.close()
        table = np.loadtxt(writename,skiprows=1, usecols=(0, 1, 2, 3, 4, 5), converters = {0: league2num, 1: datestr2num, 2:football2num, 3:football2num}, delimiter=',')
        if start:
            X = table[:, :4]
            Y = table[:, 4:]
            start=False
        else:
            X = np.append(X, table[:, :4], axis=0)
            Y = np.append(Y, table[:, 4:], axis=0)
    return data_details_return({'X': X, 'Y': Y, 'covariates': [discrete(league_dict, 'league'), datenum('match_day'), discrete(football_dict, 'home team'), discrete(football_dict, 'away team')], 'response': [integer('home score'), integer('away score')]}, data_set)
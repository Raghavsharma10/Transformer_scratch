def goodFormater(badFormat, outputPath, year, length):
    '''[summary]

    reformats the input results into a dictionary with module names as keys and their respective results as values

    outputs to csv if outputPath is specified

    Arguments:
        badFormat {dict} -- candNumber : [results for candidate]
        outputPath {str} -- the path to output to
        year {int} -- the year candidateNumber is in
        length {int} -- length of each row in badFormat divided by 2


    Returns:
        dictionary -- module : [results for module]
        saves to file if output path is specified

    '''

    devcom = 'PHAS' + badFormat['Cand'][0]

    goodFormat = {devcom: []}

    # ignore first row cause it's just 'Mark' & 'ModuleN'
    for row in list(badFormat.values())[1:]:
        goodFormat[devcom].append(int(row[0]))  # add first val to devcom

        for i in range(length-1):
            # if a key for that module doesn't exist, initialize with empt array
            goodFormat.setdefault(row[(2 * i) + 1], [])
            # add value of module to module
            goodFormat[row[(2*i)+1]].append(int(row[2*(i + 1)]))

    goodFormat.pop('0')

    goodFormat['Averages'] = everyonesAverage(year, badFormat, length)
    if outputPath is not None:  # if requested to reformat and save to file

        results = csv.writer(outputPath.open(mode='w'), delimiter=',')
        # write the keys (module names) as first row
        results.writerow(goodFormat.keys())
        # zip module results together, fill modules with less people using empty values
        # add row by row
        results.writerows(itertools.zip_longest(
            *goodFormat.values(), fillvalue=''))

    return goodFormat
def get_ranked_census_names():
    '''
    takes name lists from the the 2000 us census, prepares as a json array in order of frequency (most common names first).

    more info:
    http://www.census.gov/genealogy/www/data/2000surnames/index.html

    files in data are downloaded copies of:
    http://www.census.gov/genealogy/names/dist.all.last
    http://www.census.gov/genealogy/names/dist.male.first
    http://www.census.gov/genealogy/names/dist.female.first
    '''
    FILE_TMPL = '../data/us_census_2000_%s.txt'
    SURNAME_CUTOFF_PERCENTILE = 85 # ie7 can't handle huge lists. cut surname list off at a certain percentile.
    lists = []
    for list_name in ['surnames', 'male_first', 'female_first']:
        path = FILE_TMPL % list_name
        lst = []
        for line in codecs.open(path, 'r', 'utf8'):
            if line.strip():
                if list_name == 'surnames' and float(line.split()[2]) > SURNAME_CUTOFF_PERCENTILE:
                    break
                name = line.split()[0].lower()
                lst.append(name)
        lists.append(lst)
    return lists
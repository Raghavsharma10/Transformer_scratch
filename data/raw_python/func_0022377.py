def movie_body_count(data_set='movie_body_count'):
    """Data set of movies and body count for movies scraped from www.MovieBodyCounts.com created by Simon Garnier and Randy Olson for exploring differences between Python and R."""
    if not data_available(data_set):
        download_data(data_set)

    from pandas import read_csv
    dir_path = os.path.join(data_path, data_set)
    filename = os.path.join(dir_path, 'film-death-counts-Python.csv')
    Y = read_csv(filename)
    Y['Actors'] = Y['Actors'].apply(lambda x: x.split('|'))
    Y['Genre'] = Y['Genre'].apply(lambda x: x.split('|'))
    Y['Director'] = Y['Director'].apply(lambda x: x.split('|'))
    return data_details_return({'Y': Y, 'info' : "Data set of movies and body count for movies scraped from www.MovieBodyCounts.com created by Simon Garnier and Randy Olson for exploring differences between Python and R.",
                                }, data_set)
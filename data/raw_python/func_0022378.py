def movie_body_count_r_classify(data_set='movie_body_count'):
    """Data set of movies and body count for movies scraped from www.MovieBodyCounts.com created by Simon Garnier and Randy Olson for exploring differences between Python and R."""
    data = movie_body_count()['Y']
    import pandas as pd
    import numpy as np
    X = data[['Year', 'Body_Count']]
    Y = data['MPAA_Rating']=='R' # set label to be positive for R rated films.

    # Create series of movie genres with the relevant index
    s = data['Genre'].str.split('|').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1) # to line up with df's index

    # Extract from the series the unique list of genres.
    genres = s.unique()

    # For each genre extract the indices where it is present and add a column to X
    for genre in genres:
        index = s[s==genre].index.tolist()
        values = pd.Series(np.zeros(X.shape[0]), index=X.index)
        values[index] = 1
        X[genre] = values
    return data_details_return({'X': X, 'Y': Y, 'info' : "Data set of movies and body count for movies scraped from www.MovieBodyCounts.com created by Simon Garnier and Randy Olson for exploring differences between Python and R. In this variant we aim to classify whether the film is rated R or not depending on the genre, the years and the body count.",
                                }, data_set)
def olympic_sprints(data_set='rogers_girolami_data'):
    """All olympics sprint winning times for multiple output prediction."""
    X = np.zeros((0, 2))
    Y = np.zeros((0, 1))
    cats = {}
    for i, dataset in enumerate([olympic_100m_men,
                              olympic_100m_women,
                              olympic_200m_men,
                              olympic_200m_women,
                              olympic_400m_men,
                              olympic_400m_women]):
        data = dataset()
        year = data['X']
        time = data['Y']
        X = np.vstack((X, np.hstack((year, np.ones_like(year)*i))))
        Y = np.vstack((Y, time))
        cats[dataset.__name__] = i
    data['X'] = X
    data['Y'] = Y
    data['info'] = "Olympics sprint event winning for men and women to 2008. Data is from Rogers and Girolami's First Course in Machine Learning."
    return data_details_return({
        'X': X,
        'Y': Y,
        'covariates' : [decimalyear('year', '%Y'), discrete(cats, 'event')],
        'response' : ['time'],
        'info': "Olympics sprint event winning for men and women to 2008. Data is from Rogers and Girolami's First Course in Machine Learning.",
        'output_info': {
          0:'100m Men',
          1:'100m Women',
          2:'200m Men',
          3:'200m Women',
          4:'400m Men',
          5:'400m Women'}
        }, data_set)
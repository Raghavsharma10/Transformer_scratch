def weighted_sum(groupe, var):
    '''
    Fonction qui calcule la moyenne pondérée par groupe d'une variable
    '''
    data = groupe[var]
    weights = groupe['pondmen']
    return (data * weights).sum()
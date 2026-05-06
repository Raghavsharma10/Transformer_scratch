def gen_cannon_grad_spec(base_labels, choose, low, high, coeffs, pivots):
    """ Generate Cannon gradient spectra

    Parameters
    ----------
    labels: default values for [teff, logg, feh, cfe, nfe, afe, ak]
    choose: val of cfe or nfe, whatever you're varying
    low: lowest val of cfe or nfe, whatever you're varying
    high: highest val of cfe or nfe, whatever you're varying
    """
    # Generate Cannon gradient spectra
    low_lab = copy.copy(base_labels)
    low_lab[choose] = low
    lvec = (train_model._get_lvec(np.array([low_lab]), pivots))[0]
    model_low = np.dot(coeffs, lvec)
    high_lab = copy.copy(base_labels)
    high_lab[choose] = high
    lvec = (train_model._get_lvec(np.array([high_lab]), pivots))[0]
    model_high = np.dot(coeffs, lvec)
    grad_spec = (model_high - model_low) / (high - low)
    return grad_spec
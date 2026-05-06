def permute(num):
    "Permutation for randomizing data order."
    if permute_data:
        return np.random.permutation(num)
    else:
        logging.warning("Warning not permuting data")
        return np.arange(num)
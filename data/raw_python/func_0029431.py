def best_model(seq2hmm):
    """
    determine the best model: archaea, bacteria, eukarya (best score)
    """
    for seq in seq2hmm:
        best = []
        for model in seq2hmm[seq]:
            best.append([model, sorted([i[-1] for i in seq2hmm[seq][model]], reverse = True)[0]])
        best_model = sorted(best, key = itemgetter(1), reverse = True)[0][0]
        seq2hmm[seq] = [best_model] + [seq2hmm[seq][best_model]]
    return seq2hmm
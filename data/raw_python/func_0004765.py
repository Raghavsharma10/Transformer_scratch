def train(ds, ii):
    """ Run the training step, given a dataset object. """
    print("Loading model")
    m = model.CannonModel(2)
    print("Training...")
    m.fit(ds)
    np.savez("./ex%s_coeffs.npz" %ii, m.coeffs)
    np.savez("./ex%s_scatters.npz" %ii, m.scatters)
    np.savez("./ex%s_chisqs.npz" %ii, m.chisqs)
    np.savez("./ex%s_pivots.npz" %ii, m.pivots)
    fig = m.diagnostics_leading_coeffs(ds)
    plt.savefig("ex%s_leading_coeffs.png" %ii)
    # m.diagnostics_leading_coeffs_triangle(ds)
    # m.diagnostics_plot_chisq(ds)
    return m
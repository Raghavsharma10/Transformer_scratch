def generate_latex_table(sampled_hmm, conf=0.95, dt=1, time_unit='ms', obs_name='force', obs_units='pN',
                         caption='', outfile=None):
    """
    Generate a LaTeX column-wide table showing various computed properties and uncertainties.

    Parameters
    ----------
    conf : float
        confidence interval. Use 0.68 for 1 sigma, 0.95 for 2 sigma etc.

    """
    # check input
    from bhmm.hmm.generic_sampled_hmm import SampledHMM
    from bhmm.hmm.gaussian_hmm import SampledGaussianHMM
    assert issubclass(sampled_hmm.__class__, SampledHMM), 'sampled_hmm ist not a SampledHMM'

    # confidence interval
    sampled_hmm.set_confidence(conf)
    # dt
    dt = float(dt)
    # nstates
    nstates = sampled_hmm.nstates

    table = """
\\begin{table}
    \\begin{tabular*}{\columnwidth}{@{\extracolsep{\\fill}}lcc}
        \hline
        {\\bf Property} & {\\bf Symbol} & {\\bf Value} \\\\
        \hline
            """
    # Stationary probability.
    p = sampled_hmm.stationary_distribution_mean
    p_lo, p_hi = sampled_hmm.stationary_distribution_conf
    for i in range(nstates):
        if i == 0:
            table += '\t\tEquilibrium probability '
        table += '\t\t& $\pi_{%d}$ & $%0.3f_{\:%0.3f}^{\:%0.3f}$ \\\\' % (i+1, p[i], p_lo[i], p_hi[i]) + '\n'
    table += '\t\t\hline' + '\n'

    # Transition probabilities.
    P = sampled_hmm.transition_matrix_mean
    P_lo, P_hi = sampled_hmm.transition_matrix_conf
    for i in range(nstates):
        for j in range(nstates):
            if i == 0 and j == 0:
                table += '\t\tTransition probability ($\Delta t = $%s) ' % (str(dt)+' '+time_unit)
            table += '\t\t& $T_{%d%d}$ & $%0.4f_{\:%0.4f}^{\:%0.4f}$ \\\\' % (i+1, j+1, P[i, j], P_lo[i, j], P_hi[i, j]) + '\n'
    table += '\t\t\hline' + '\n'
    table += '\t\t\hline' + '\n'

    # Transition rates via pseudogenerator.
    K = P - np.eye(sampled_hmm.nstates)
    K /= dt
    K_lo = P_lo - np.eye(sampled_hmm.nstates)
    K_lo /= dt
    K_hi = P_hi - np.eye(sampled_hmm.nstates)
    K_hi /= dt
    for i in range(nstates):
        for j in range(nstates):
            if i == 0 and j == 0:
                table += '\t\tTransition rate (%s$^{-1}$) ' % time_unit
            if i != j:
                table += '\t\t& $k_{%d%d}$ & $%2.4f_{\:%2.4f}^{\:%2.4f}$ \\\\' % (i+1, j+1, K[i, j], K_lo[i, j], K_hi[i, j]) + '\n'
    table += '\t\t\hline' + '\n'

    # State mean lifetimes.
    l = sampled_hmm.lifetimes_mean
    l *= dt
    l_lo, l_hi = sampled_hmm.lifetimes_conf
    l_lo *= dt
    l_hi *= dt
    for i in range(nstates):
        if i == 0:
            table += '\t\tState mean lifetime (%s) ' % time_unit
        table += '\t\t& $t_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\\\' % (i+1, l[i], l_lo[i], l_hi[i]) + '\n'
    table += '\t\t\hline' + '\n'

    # State relaxation timescales.
    t = sampled_hmm.timescales_mean
    t *= dt
    t_lo, t_hi = sampled_hmm.timescales_conf
    t_lo *= dt
    t_hi *= dt
    for i in range(nstates-1):
        if i == 0:
            table += '\t\tRelaxation time (%s) ' % time_unit
        table += '\t\t& $\\tau_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\\\' % (i+1, t[i], t_lo[i], t_hi[i]) + '\n'
    table += '\t\t\hline' + '\n'

    if issubclass(sampled_hmm.__class__, SampledGaussianHMM):
        table += '\t\t\hline' + '\n'

        # State mean forces.
        m = sampled_hmm.means_mean
        m_lo, m_hi = sampled_hmm.means_conf
        for i in range(nstates):
            if i == 0:
                table += '\t\tState %s mean (%s) ' % (obs_name, obs_units)
            table += '\t\t& $\mu_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\\\' % (i+1, m[i], m_lo[i], m_hi[i]) + '\n'
        table += '\t\t\hline' + '\n'

        # State force standard deviations.
        s = sampled_hmm.sigmas_mean
        s_lo, s_hi = sampled_hmm.sigmas_conf
        for i in range(nstates):
            if i == 0:
                table += '\t\tState %s std dev (%s) ' % (obs_name, obs_units)
            table += '\t\t& $s_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\\\' % (i+1, s[i], s_lo[i], s_hi[i]) + '\n'
        table += '\t\t\hline' + '\n'

    table += """
        \\hline
    \\end{tabular*}
    \\caption{{\\bf %s}}
\\end{table}
            """ % caption

    # write to file if wanted:
    if outfile is not None:
        f = open(outfile, 'w')
        f.write(table)
        f.close()

    return table
def guess_geometry(graph, unit_cell=None, verbose=False):
    """Construct a molecular geometry based on a molecular graph.

       This routine does not require initial coordinates and will give a very
       rough picture of the initial geometry. Do not expect all details to be
       in perfect condition. A subsequent optimization with a more accurate
       level of theory is at least advisable.

       Argument:
        | ``graph``  --  The molecular graph of the system, see
                         :class:molmod.molecular_graphs.MolecularGraph

       Optional argument:
        | ``unit_cell``  --  periodic boundry conditions, see
                             :class:`molmod.unit_cells.UnitCell`
        | ``verbose``  --  Show optimizer progress when True
    """

    N = len(graph.numbers)
    from molmod.minimizer import Minimizer, ConjugateGradient, \
        NewtonLineSearch, ConvergenceCondition, StopLossCondition

    search_direction = ConjugateGradient()
    line_search = NewtonLineSearch()
    convergence = ConvergenceCondition(grad_rms=1e-6, step_rms=1e-6)
    stop_loss = StopLossCondition(max_iter=500, fun_margin=0.1)

    ff = ToyFF(graph, unit_cell)
    x_init = np.random.normal(0, 1, N*3)

    #  level 1 geometry optimization: graph based
    ff.dm_quad = 1.0
    minimizer = Minimizer(x_init, ff, search_direction, line_search, convergence, stop_loss, anagrad=True, verbose=verbose)
    x_init = minimizer.x

    #  level 2 geometry optimization: graph based + pauli repulsion
    ff.dm_quad = 1.0
    ff.dm_reci = 1.0
    minimizer = Minimizer(x_init, ff, search_direction, line_search, convergence, stop_loss, anagrad=True, verbose=verbose)
    x_init = minimizer.x

    # Add a little noise to avoid saddle points
    x_init += np.random.uniform(-0.01, 0.01, len(x_init))

    #  level 3 geometry optimization: bond lengths + pauli
    ff.dm_quad = 0.0
    ff.dm_reci = 0.2
    ff.bond_quad = 1.0
    minimizer = Minimizer(x_init, ff, search_direction, line_search, convergence, stop_loss, anagrad=True, verbose=verbose)
    x_init = minimizer.x

    #  level 4 geometry optimization: bond lengths + bending angles + pauli
    ff.bond_quad = 0.0
    ff.bond_hyper = 1.0
    ff.span_quad = 1.0
    minimizer = Minimizer(x_init, ff, search_direction, line_search, convergence, stop_loss, anagrad=True, verbose=verbose)
    x_init = minimizer.x

    x_opt = x_init

    mol = Molecule(graph.numbers, x_opt.reshape((N, 3)))
    return mol
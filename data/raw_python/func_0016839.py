def rime_solver(slvr_cfg):
    """ Factory function that produces a RIME solver """
    from montblanc.impl.rime.tensorflow.RimeSolver import RimeSolver
    return RimeSolver(slvr_cfg)
def plot_iv_curve(a, hold_v, i, *plt_args, **plt_kwargs):    
    """A single IV curve"""
    grid = plt_kwargs.pop('grid',True)
    same_fig = plt_kwargs.pop('same_fig',False) 
    if not len(plt_args):
        plt_args = ('ko-',)
    if 'label' not in plt_kwargs: 
        plt_kwargs['label'] = 'Current'
    if not same_fig:
        make_iv_curve_fig(a, grid=grid)
    if type(i) is dict:
        i = [i[v] for v in hold_v]
    plt.plot([v*1e3 for v in hold_v], [ii*1e12 for ii in i], *plt_args, **plt_kwargs)
    plt.legend(loc=2)
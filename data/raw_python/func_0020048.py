def EverestModel(ID, model='nPLD', publish=False, csv=False, **kwargs):
    '''
    A wrapper around an :py:obj:`everest` model for PBS runs.

    '''

    if model != 'Inject':
        from ... import detrender

        # HACK: We need to explicitly mask short cadence planets
        if kwargs.get('cadence', 'lc') == 'sc':
            EPIC, t0, period, duration = \
                np.loadtxt(os.path.join(EVEREST_SRC, 'missions', 'k2',
                                        'tables', 'scmasks.tsv'), unpack=True)
            if ID in EPIC and kwargs.get('planets', None) is None:
                ii = np.where(EPIC == ID)[0]
                planets = []
                for i in ii:
                    planets.append([t0[i], period[i], 1.25 * duration[i]])
                kwargs.update({'planets': planets})

        # Run the model
        m = getattr(detrender, model)(ID, **kwargs)

        # Publish?
        if publish:
            if csv:
                m.publish_csv()
            else:
                m.publish()

    else:
        from ...inject import Inject
        Inject(ID, **kwargs)
    return True
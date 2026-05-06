def prepare_full_example_2(lastdate='1996-01-05') -> (
        hydpytools.HydPy, hydpy.pub, testtools.TestIO):
    """Prepare the complete `LahnH` project for testing.

    |prepare_full_example_2| calls |prepare_full_example_1|, but also
    returns a readily prepared |HydPy| instance, as well as module
    |pub| and class |TestIO|, for convenience:

    >>> from hydpy.core.examples import prepare_full_example_2
    >>> hp, pub, TestIO = prepare_full_example_2()
    >>> hp.nodes
    Nodes("dill", "lahn_1", "lahn_2", "lahn_3")
    >>> hp.elements
    Elements("land_dill", "land_lahn_1", "land_lahn_2", "land_lahn_3",
             "stream_dill_lahn_2", "stream_lahn_1_lahn_2",
             "stream_lahn_2_lahn_3")
    >>> pub.timegrids
    Timegrids(Timegrid('1996-01-01 00:00:00',
                       '1996-01-05 00:00:00',
                       '1d'))
    >>> from hydpy import classname
    >>> classname(TestIO)
    'TestIO'

    The last date of the initialisation period is configurable:

    >>> hp, pub, TestIO = prepare_full_example_2('1996-02-01')
    >>> pub.timegrids
    Timegrids(Timegrid('1996-01-01 00:00:00',
                       '1996-02-01 00:00:00',
                       '1d'))
    """
    prepare_full_example_1()
    with testtools.TestIO():
        hp = hydpytools.HydPy('LahnH')
        hydpy.pub.timegrids = '1996-01-01', lastdate, '1d'
        hp.prepare_everything()
    return hp, hydpy.pub, testtools.TestIO
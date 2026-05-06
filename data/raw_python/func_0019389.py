def prepare_full_example_1() -> None:
    """Prepare the complete `LahnH` project for testing.

    >>> from hydpy.core.examples import prepare_full_example_1
    >>> prepare_full_example_1()
    >>> from hydpy import TestIO
    >>> import os
    >>> with TestIO():
    ...     print('root:', *sorted(os.listdir('.')))
    ...     for folder in ('control', 'conditions', 'series'):
    ...         print(f'LahnH/{folder}:',
    ...               *sorted(os.listdir(f'LahnH/{folder}')))
    root: LahnH __init__.py
    LahnH/control: default
    LahnH/conditions: init_1996_01_01
    LahnH/series: input node output temp

    ToDo: Improve, test, and explain this example data set.
    """
    testtools.TestIO.clear()
    shutil.copytree(
        os.path.join(data.__path__[0], 'LahnH'),
        os.path.join(iotesting.__path__[0], 'LahnH'))
    seqpath = os.path.join(iotesting.__path__[0], 'LahnH', 'series')
    for folder in ('output', 'node', 'temp'):
        os.makedirs(os.path.join(seqpath, folder))
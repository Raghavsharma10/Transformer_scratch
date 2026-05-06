def _add_lines(specification, module):
    """Return autodoc commands for a basemodels docstring.

    Note that `collection classes` (e.g. `Model`, `ControlParameters`,
    `InputSequences` are placed on top of the respective section and the
    `contained classes` (e.g. model methods, `ControlParameter` instances,
    `InputSequence` instances at the bottom.  This differs from the order
    of their definition in the respective modules, but results in a better
    documentation structure.
    """
    caption = _all_spec2capt.get(specification, 'dummy')
    if caption.split()[-1] in ('parameters', 'sequences', 'Masks'):
        exists_collectionclass = True
        name_collectionclass = caption.title().replace(' ', '')
    else:
        exists_collectionclass = False
    lines = []
    if specification == 'model':
        lines += [f'',
                  f'.. autoclass:: {module.__name__}.Model',
                  f'    :members:',
                  f'    :show-inheritance:',
                  f'    :exclude-members: {", ".join(EXCLUDE_MEMBERS)}']
    elif exists_collectionclass:
        lines += [f'',
                  f'.. autoclass:: {module.__name__}.{name_collectionclass}',
                  f'    :members:',
                  f'    :show-inheritance:',
                  f'    :exclude-members: {", ".join(EXCLUDE_MEMBERS)}']
    lines += ['',
              '.. automodule:: ' + module.__name__,
              '    :members:',
              '    :show-inheritance:']
    if specification == 'model':
        lines += ['    :exclude-members: Model']
    elif exists_collectionclass:
        lines += ['    :exclude-members: ' + name_collectionclass]
    return lines
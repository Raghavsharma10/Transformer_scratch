def reverse_model_wildcard_import():
    """Clear the local namespace from a model wildcard import.

    Calling this method should remove the critical imports into the local
    namespace due the last wildcard import of a certain application model.
    It is thought for securing the successive preperation of different
    types of models via wildcard imports.  See the following example, on
    how it can be applied.

    >>> from hydpy import reverse_model_wildcard_import

    Assume you wildcard import the first version of HydPy-L-Land (|lland_v1|):

    >>> from hydpy.models.lland_v1 import *

    This for example adds the collection class for handling control
    parameters of `lland_v1` into the local namespace:

    >>> print(ControlParameters(None).name)
    control

    Calling function |parameterstep| for example prepares the control
    parameter object |lland_control.NHRU|:

    >>> parameterstep('1d')
    >>> nhru
    nhru(?)

    Calling function |reverse_model_wildcard_import| removes both
    objects (and many more, but not all) from the local namespace:

    >>> reverse_model_wildcard_import()

    >>> ControlParameters
    Traceback (most recent call last):
    ...
    NameError: name 'ControlParameters' is not defined

    >>> nhru
    Traceback (most recent call last):
    ...
    NameError: name 'nhru' is not defined
    """
    namespace = inspect.currentframe().f_back.f_locals
    model = namespace.get('model')
    if model is not None:
        for subpars in model.parameters:
            for par in subpars:
                namespace.pop(par.name, None)
                namespace.pop(objecttools.classname(par), None)
            namespace.pop(subpars.name, None)
            namespace.pop(objecttools.classname(subpars), None)
        for subseqs in model.sequences:
            for seq in subseqs:
                namespace.pop(seq.name, None)
                namespace.pop(objecttools.classname(seq), None)
            namespace.pop(subseqs.name, None)
            namespace.pop(objecttools.classname(subseqs), None)
        for name in ('parameters', 'sequences', 'masks', 'model',
                     'Parameters', 'Sequences', 'Masks', 'Model',
                     'cythonizer', 'cymodel', 'cythonmodule'):
            namespace.pop(name, None)
        for key in list(namespace.keys()):
            try:
                if namespace[key].__module__ == model.__module__:
                    del namespace[key]
            except AttributeError:
                pass
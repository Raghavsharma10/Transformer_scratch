def PatchAt(cls, n, module, method_wrapper=None, module_alias=None, method_name_modifier=utils.identity, blacklist_predicate=_False, whitelist_predicate=_True, return_type_predicate=_None, getmembers_predicate=inspect.isfunction, admit_private=False, explanation=""):
        """
This classmethod lets you easily patch all of functions/callables from a module or class as methods a Builder class.

**Arguments**

* **n** : the position the the object being piped will take in the arguments when the function being patched is applied. See `RegisterMethod` and `ThenAt`.
* **module** : a module or class from which the functions/methods/callables will be taken.
* `module_alias = None` : an optional alias for the module used for documentation purposes.
* `method_name_modifier = lambda f_name: None` : a function that can modify the name of the method will take. If `None` the name of the function will be used.
* `blacklist_predicate = lambda f_name: name[0] != "_"` : A predicate that determines which functions are banned given their name. By default it excludes all function whose name start with `'_'`. `blacklist_predicate` can also be of type list, in which case all names contained in this list will be banned.
* `whitelist_predicate = lambda f_name: True` : A predicate that determines which functions are admitted given their name. By default it include any function. `whitelist_predicate` can also be of type list, in which case only names contained in this list will be admitted. You can use both `blacklist_predicate` and `whitelist_predicate` at the same time.
* `return_type_predicate = lambda f_name: None` : a predicate that determines the `_return_type` of the Builder. By default it will always return `None`. See `phi.builder.Builder.ThenAt`.
* `getmembers_predicate = inspect.isfunction` : a predicate that determines what type of elements/members will be fetched by the `inspect` module, defaults to [inspect.isfunction](https://docs.python.org/2/library/inspect.html#inspect.isfunction). See [getmembers](https://docs.python.org/2/library/inspect.html#inspect.getmembers).

**Examples**

Lets patch ALL the main functions from numpy into a custom builder!

    from phi import PythonBuilder #or Builder
    import numpy as np

    class NumpyBuilder(PythonBuilder): #or Builder
        "A Builder for numpy functions!"
        pass

    NumpyBuilder.PatchAt(1, np)

    N = NumpyBuilder(lambda x: x)

Thats it! Although a serious patch would involve filtering out functions that don't take arrays. Another common task would be to use `NumpyBuilder.PatchAt(2, ...)` (`PatchAt(n, ..)` in general) when convenient to send the object being pipe to the relevant argument of the function. The previous is usually done with and a combination of `whitelist_predicate`s and `blacklist_predicate`s on `PatchAt(1, ...)` and `PatchAt(2, ...)` to filter or include the approriate functions on each kind of patch. Given the previous code we could now do

    import numpy as np

    x = np.array([[1,2],[3,4]])
    y = np.array([[5,6],[7,8]])

    z = N.Pipe(
        x, N
        .dot(y)
        .add(x)
        .transpose()
        .sum(axis=1)
    )

Which is strictly equivalent to

    import numpy as np

    x = np.array([[1,2],[3,4]])
    y = np.array([[5,6],[7,8]])

    z = np.dot(x, y)
    z = np.add(z, x)
    z = np.transpose(z)
    z = np.sum(z, axis=1)

The thing to notice is that with the `NumpyBuilder` we avoid the repetitive and needless passing and reassigment of the `z` variable, this removes a lot of noise from our code.
        """
        _rtp = return_type_predicate

        return_type_predicate = (lambda x: _rtp) if inspect.isclass(_rtp) and issubclass(_rtp, Builder) else _rtp
        module_name = module_alias if module_alias else module.__name__ + '.'
        patch_members = _get_patch_members(module, blacklist_predicate=blacklist_predicate, whitelist_predicate=whitelist_predicate, getmembers_predicate=getmembers_predicate, admit_private=admit_private)

        for name, f in patch_members:
            wrapped = None

            if method_wrapper:
                g = method_wrapper(f)
                wrapped = f
            else:
                g = f

            cls.RegisterAt(n, g, module_name, wrapped=wrapped, _return_type=return_type_predicate(name), alias=method_name_modifier(name), explanation=explanation)
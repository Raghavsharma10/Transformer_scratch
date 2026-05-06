def _hash_filter_fn(self, filter_fn, **kwargs):
        """ Construct string representing state of filter_fn
            Used to cache filtered variants or effects uniquely depending on filter fn values
        """
        filter_fn_name = self._get_function_name(filter_fn, default="filter-none")
        logger.debug("Computing hash for filter_fn: {} with kwargs {}".format(filter_fn_name, str(dict(**kwargs))))
        # hash function source code
        fn_source = str(dill.source.getsource(filter_fn))
        pickled_fn_source = pickle.dumps(fn_source) ## encode as byte string
        hashed_fn_source = int(hashlib.sha1(pickled_fn_source).hexdigest(), 16) % (10 ** 11)
        # hash kwarg values
        kw_dict = dict(**kwargs)
        kw_hash = list()
        if not kw_dict:
            kw_hash = ["default"]
        else:
            [kw_hash.append("{}-{}".format(key, h)) for (key, h) in sorted(kw_dict.items())]
        # hash closure vars - for case where filter_fn is defined within closure of filter_fn
        closure = []
        nonlocals = inspect.getclosurevars(filter_fn).nonlocals
        for (key, val) in nonlocals.items():
            ## capture hash for any function within closure
            if inspect.isfunction(val):
                closure.append(self._hash_filter_fn(val))
        closure.sort() # Sorted for file name consistency
        closure_str = "null" if len(closure) == 0 else "-".join(closure)
        # construct final string comprising hashed components
        hashed_fn = ".".join(["-".join([filter_fn_name,
                                        str(hashed_fn_source)]),
                              ".".join(kw_hash),
                              closure_str]
                            )
        return hashed_fn
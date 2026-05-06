def get(cls, route, args, kwargs):
        """
        :cls        (class) <Class> of the object requested
        :route      (dict) Debris route schema
        :args       (list) of argument provided for initializing
        :kwargs     (dict) of given data to construct the object
        """
        # ---------
        # Namespace
        # ---------
        if None in args:
            namespace = None
        else:
            namespace = call(route.get('namespace'), args, kwargs) if route.get('namespace') \
                        else ".".join(map(str, [cls.__name__] + list(args)))

        # bool, can store in memory
        # _in_memory = route.get('memory', True)

        # ---------------
        # Get from Memory
        # ---------------
        # if _in_memory and namespace:
        # check for this namespace
        try:
            return debris.services.memory.get(namespace)
        except LookupError:
            pass

        # ------------------------
        # Constructed w/ init args
        # ------------------------
        # - this method bypasses many of the debris
        #   features for cacheing because the 
        #   kwargs should contain all the construction information
        # - replaces existing object in memory beceause this data is 
        #   given to be "newer" data
        if len(kwargs) > 0:
            cls = call(route.get('substitute'), args, kwargs) or cls
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            if namespace:
                debris.services.memory.set(namespace, obj)
            return obj

        elif not namespace:
            raise LookupError("No id/key provided to initialize object "+namespace.replace('.', '(', 1).replace('.', ', ')+")")

        # -----------------
        # Retrieve the Data
        # -----------------
        insp = inspect.getargspec(cls.__init__)
        data = None
        if route.get('get'):
            for r in route['get']:
                if r['service'] == 'postgresql':
                    iwargs = dict([(k, args[i] if len(args) > i else None) for i, k in enumerate(insp.args[1:])])
                    data = r["bank"].get(r['query'], **iwargs)
                else:
                    data = r["bank"].get(namespace)
                if data:
                    break

            # -----------------
            # Retrieve the Data
            # -----------------
            if not data:
                raise LookupError("Data could not be found for "+namespace.replace('.', '(', 1).replace('.', ', ')+")")

        if not data:
            data = {}

        # --------------------
        # Manage Args / Kwargs
        # --------------------
        # remove the default "self" argument
        insp.args.pop(0)
        [data.pop(k) for k in insp.args if k in data]

        # substiture class w/ known data
        if route.get('substitute'):
            cls = callattr(cls, route['substitute'], args, data) or cls

        # ----------------
        # Initialize Class
        # ----------------
        obj = cls.__new__(cls, *args, **data)
        obj.__init__(*args, **data)

        # ---------------
        # Store in Memory
        # ---------------
        if namespace:
            debris.services.memory.set(namespace, obj)

        # return the constructed object
        return obj
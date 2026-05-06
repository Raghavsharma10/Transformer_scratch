def getmany(cls, route, args, kwargs, _keys):
        """
        1. build name space
        2. look locally for copies
        3. build group for batch
        4. fetch the new ones
        5. return found + new list
        """
        # copy the list of keys
        keys = [] + _keys
        # build a list of returning objects
        returning = []
        # dictionary of references
        namespaces = {} # key: ns
        namespace_keys = {} # ns: key
        
        # shorthand
        returning_append = returning.append
        memory_get = debris.services.memory.get
        memory_set = debris.services.memory.set

        # ---------------
        # Get from Memory
        # ---------------
        for key in keys:
            # ---------
            # Namespace
            # ---------
            namespace = call(route.get('namespace'), args + [key], kwargs) if route.get('namespace') \
                        else ".".join(map(str, [cls.__name__] + args + [key]))
            
            # check for this namespace
            try:
                returning_append(memory_get(namespace))
            except LookupError:
                # not found, add to namespace list
                namespaces[str(key)] = namespace
                namespace_keys[namespace] = str(key)

        if not namespaces:
            # all data found, return the findings
            return returning

        # -----------------
        # Retrieve the Data
        # -----------------
        insp = inspect.getargspec(cls.__init__)
        insp.args.pop(0) # self
        data = None
        for r in route['get']:
            if r['service'] == 'postgresql':
                iwargs = dict([(k, args[i] if len(args) > i else None) for i, k in enumerate(insp.args[:-1])])
                iwargs[insp.args[-1]] = namespaces.keys()
                # create a limit, speed up the query
                iwargs['limit'] = len(namespaces)
                results = r["bank"].getmany(r['query[]'], **iwargs)
                if results:
                    # retrieve the key from the results. hacky way, but works
                    key = insp.args[-1]
                    # pop out the "key" for each row, ex. "id", then switch to the namespace
                    # keys[row.pop('id')] => "user.1"
                    results = [(namespaces[str(row[key])], row) for row in results]

            else:
                results = r["bank"].getmany(namespaces.values())

            # Results Found
            # -------------
            if results:
                # [(ns, data), ...]
                for namespace, data in results:
                    if data:
                        # substiture class w/ known data
                        # ------------------------------
                        if route.get('substitute'):
                            _cls = callattr(cls, route.get('substitute'), args, data) or cls
                        else:
                            _cls = cls

                        # initialize class
                        # ----------------
                        obj = _cls.__new__(_cls, *args, **data)
                        obj.__init__(*args, **data)

                        # store in memory
                        # ---------------
                        memory_set(namespace, obj)
                        namespaces.pop(namespace_keys.pop(namespace))

                        # add the constructed object
                        # --------------------------
                        returning_append(obj)

        return returning
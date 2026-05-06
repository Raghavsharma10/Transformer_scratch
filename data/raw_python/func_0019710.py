def order(self, egg, method='permute', nperms=2500, strategy=None,
              distfun='correlation', fingerprint=None):
        """
        Reorders a list of stimuli to match a fingerprint

        Parameters
        ----------
        egg : quail.Egg
            Data to compute fingerprint

        method : str
            Method to re-sort list. Can be 'stick' or 'permute' (default: permute)

        nperms : int
            Number of permutations to use. Only used if method='permute'. (default:
            2500)

        strategy : str or None
            The strategy to use to reorder the list.  This can be 'stabilize',
            'destabilize', 'random' or None.  If None, the self.strategy field
            will be used. (default: None)

        distfun : str or function
            The distance function to reorder the list fingerprint to the target
            fingerprint.  Can be any distance function supported by
            scipy.spatial.distance.cdist. For more info, see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
            (default: euclidean)

        fingerprint : quail.Fingerprint or np.array
            Fingerprint (or just the state of a fingerprint) to reorder by. If
            None, the list will be reordered according to the fingerprint
            attached to the presenter object.

        Returns
        ----------
        egg : quail.Egg
            Egg re-sorted to match fingerprint
        """

        def order_perm(self, egg, dist_dict, strategy, nperm, distperm,
                       fingerprint):
            """
            This function re-sorts a list by computing permutations of a given
            list and choosing the one that maximizes/minimizes variance.
            """

            # parse egg
            pres, rec, features, dist_funcs = parse_egg(egg)

            # length of list
            pres_len = len(pres)

            weights = []
            orders = []
            for i in range(nperms):
                x = rand_perm(pres, features, dist_dict, dist_funcs)
                weights.append(x[0])
                orders.append(x[1])
            weights = np.array(weights)
            orders = np.array(orders)

            # find the closest (or farthest)
            if strategy=='stabilize':
                closest = orders[np.nanargmin(cdist(np.array(fingerprint, ndmin=2), weights, distperm)),:].astype(int).tolist()
            elif strategy=='destabilize':
                closest = orders[np.nanargmax(cdist(np.array(fingerprint, ndmin=2), weights, distperm)),:].astype(int).tolist()

            # return a re-sorted egg
            return Egg(pres=[list(pres[closest])], rec=[list(pres[closest])], features=[list(features[closest])])

        def order_best_stick(self, egg, dist_dict, strategy, nperms, distfun,
                             fingerprint):

            # parse egg
            pres, rec, features, dist_funcs = parse_egg(egg)

            results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(stick_perm)(self, egg, dist_dict, strategy) for i in range(nperms))

            weights = np.array([x[0] for x in results])
            orders = np.array([x[1] for x in results])

            # find the closest (or farthest)
            if strategy=='stabilize':
                closest = orders[np.nanargmin(cdist(np.array(fingerprint, ndmin=2), weights, distfun)),:].astype(int).tolist()
            elif strategy=='destabilize':
                closest = orders[np.nanargmax(cdist(np.array(fingerprint, ndmin=2), weights, distfun)),:].astype(int).tolist()

            # return a re-sorted egg
            return Egg(pres=[list(pres[closest])], rec=[list(pres[closest])], features=[list(features[closest])], dist_funcs=dist_funcs)

        def order_best_choice(self, egg, dist_dict, nperms, distfun,
                              fingerprint):

            # get strategy
            strategy = self.strategy

            # parse egg
            pres, rec, features, dist_funcs = parse_egg(egg)

            results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(choice_perm)(self, egg, dist_dict) for i in range(nperms))

            weights = np.array([x[0] for x in results])
            orders = np.array([x[1] for x in results])

            # find the closest (or farthest)
            if strategy=='stabilize':
                closest = orders[np.nanargmin(cdist(np.array(fingerprint, ndmin=2), weights, distfun)),:].astype(int).tolist()
            elif strategy=='destabilize':
                closest = orders[np.nanargmax(cdist(np.array(fingerprint, ndmin=2), weights, distfun)),:].astype(int).tolist()

            # return a re-sorted egg
            return Egg(pres=[list(pres[closest])], rec=[list(pres[closest])], features=[list(features[closest])], dist_funcs=dist_funcs)

        # if strategy is not set explicitly, default to the class strategy
        if strategy is None:
            strategy = self.strategy

        dist_dict = compute_distances_dict(egg)

        if fingerprint is None:
            fingerprint = self.get_params('fingerprint').state
        elif isinstance(fingerprint, Fingerprint):
            fingerprint = fingerprint.state
        else:
            print('using custom fingerprint')

        if (strategy=='random') or (method=='random'):
            return shuffle_egg(egg)
        elif method=='permute':
            return order_perm(self, egg, dist_dict, strategy, nperms, distfun,
                              fingerprint) #
        elif method=='stick':
            return order_stick(self, egg, dist_dict, strategy, fingerprint) #
        elif method=='best_stick':
            return order_best_stick(self, egg, dist_dict, strategy, nperms,
                                    distfun, fingerprint) #
        elif method=='best_choice':
            return order_best_choice(self, egg, dist_dict, nperms,
                                     fingerprint)
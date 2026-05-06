def fancy(cls, contains, max_tries, inner=False, keepcase=False):
        """
        Try to create a key with a chosen prefix, by starting with a 26-bit
        urandom number and appending with 8-byte integers until prefix matches.
        This function is naive, but has a max_tries argument which will abort when
        reached with a ValueError.
        TODO: make this smarter, in general. Variable byte length according to
        expected attempts, warnings of expected duration of iteration, etc. 
        TODO: Implement multiprocessing to use poly-core machines fully:
            - Shared list, each process checks if empty every cycle, aborts if
              contains a value.
            - Successful values are pushed to list, cancelling all processes?
            - Server waits on all child processes then expects a list?
            - Ensure child processes start with different random base numbers,
              to avoid duplication?
            - Investigate server/manager aspect of multiprocessing; mini-clustering?
        """
        contains = contains if keepcase else contains.lower()
        if not set(contains).issubset(base58.alphabet):
            raise ValueError("Cannot find contained phrase '{}' as it contains non-b58 characters".format(contains))
        basenum = os.urandom(26)
        for i in range(max_tries):
            k = nacl.public.PrivateKey(basenum + i.to_bytes(6, 'big'))
            ukey = cls(k.public_key, k)
            test_uid = ukey.userID if keepcase else ukey.userID.lower()
            if test_uid.startswith(contains) or test_uid.endswith(contains) or (inner and contains in test_uid):
                return ukey
        else:
            raise ValueError("Could not create key with desired prefix '{}' in {} attempts.".format(prefix, max_tries))
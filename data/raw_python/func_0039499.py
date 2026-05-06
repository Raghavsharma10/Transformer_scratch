def frog_tip(self):
        """\
        Return a single FROG tip.
        """
        cache = self._cache
        client = self._client

        if self.should_refresh:
            tips = client.croak()
            for number, tip in tips.items():
                cache[str(number)] = tip

        choice = random.choice(list(cache.keys()))

        # We'll get a bytes() object here during real usage
        # but a text-like object in the tests. Good job Python
        try:
            tip = cache[choice].decode()
        except AttributeError:
            tip = cache[choice]

        del cache[choice]

        return tip
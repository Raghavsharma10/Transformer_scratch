def simulate(self):
        """Simulates a stream of types."""
        # Simulates zero to 10 types
        return [t.simulate() for t in itertools.islice(self, random.choice(range(10)))]
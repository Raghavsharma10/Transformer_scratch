def crossover_template(cls, length, points=2):
        """Create a crossover template with the given number of points. The
        crossover template can be used as a mask to crossover two
        bitstrings of the same length.

        Usage:
            assert len(parent1) == len(parent2)
            template = BitString.crossover_template(len(parent1))
            inv_template = ~template
            child1 = (parent1 & template) | (parent2 & inv_template)
            child2 = (parent1 & inv_template) | (parent2 & template)

        Arguments:
            length: An int, indicating the desired length of the result.
            points: An int, the number of crossover points.
        Return:
            A BitString instance of the requested length which can be used
            as a crossover template.
        """

        assert isinstance(length, int) and length >= 0
        assert isinstance(points, int) and points >= 0

        # Select the crossover points.
        points = random.sample(range(length + 1), points)

        # Prep the points for the loop.
        points.sort()
        points.append(length)

        # Fill the bits in with alternating ranges of 0 and 1 according to
        # the selected crossover points.
        previous = 0
        include_range = bool(random.randrange(2))
        bits = 0
        for point in points:
            if point > previous:
                bits <<= point - previous
                if include_range:
                    bits += (1 << (point - previous)) - 1
            include_range = not include_range
            previous = point

        return cls(bits, length)
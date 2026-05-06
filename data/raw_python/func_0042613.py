def weight(weights):
        """
        RETURN RANDOM INDEX INTO WEIGHT ARRAY, GIVEN WEIGHTS
        """
        total = sum(weights)

        p = SEED.random()
        acc = 0
        for i, w in enumerate(weights):
            acc += w / total
            if p < acc:
                return i
        return len(weights) - 1
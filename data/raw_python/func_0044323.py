def check_move(new, old, t):
        """Determines if a model will be accepted."""
        if (t <= 0) or numpy.isclose(t, 0.0):
            return False
        K_BOLTZ = 1.9872041E-003  # kcal/mol.K
        if new < old:
            return True
        else:
            move_prob = math.exp(-(new - old) / (K_BOLTZ * t))
            if move_prob > random.uniform(0, 1):
                return True
        return False
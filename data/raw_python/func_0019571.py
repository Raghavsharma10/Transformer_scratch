def randomize(self):
        """Create a new motif with shuffled positions.

        Shuffle the positions of this motif and return a new Motif instance.

        Returns
        -------
        m : Motif instance
            Motif instance with shuffled positions.
        """
        random_pfm = [[c for c in row] for row in self.pfm]
        random.shuffle(random_pfm)
        m = Motif(pfm=random_pfm)
        m.id = "random"
        return m
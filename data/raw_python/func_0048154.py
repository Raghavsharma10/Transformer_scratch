def generate_one(self):
        """Generate a single element.
    
        Returns
        -------
        element
            An element from the domain.
    
    
        Examples
        -------
        >>> generator = RepellentGenerator(['a', 'b'])
        >>> gen_item = generator.generate_one()
        >>> gen_item in ['a', 'b']
        True
        """
        # Get the weights for all items in the domain
        weights = [self.probability_func(self.generated[element])
                   for element in self.domain]

        # Sample from the domain using the weights
        element = random.choices(self.domain, weights=weights)[0]

        # Update the generated values and return
        self.generated[element] += 1

        return element
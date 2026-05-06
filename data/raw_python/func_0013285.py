def generateVariant(self, referenceName, position, randomNumberGenerator):
        """
        Generate a random variant for the specified position using the
        specified random number generator. This generator should be seeded
        with a value that is unique to this position so that the same variant
        will always be produced regardless of the order it is generated in.
        """
        variant = self._createGaVariant()
        variant.reference_name = referenceName
        variant.start = position
        variant.end = position + 1  # SNPs only for now
        bases = ["A", "C", "G", "T"]
        ref = randomNumberGenerator.choice(bases)
        variant.reference_bases = ref
        alt = randomNumberGenerator.choice(
            [base for base in bases if base != ref])
        variant.alternate_bases.append(alt)
        randChoice = randomNumberGenerator.randint(0, 2)
        if randChoice == 0:
            variant.filters_applied = False
        elif randChoice == 1:
            variant.filters_applied = True
            variant.filters_passed = True
        else:
            variant.filters_applied = True
            variant.filters_passed = False
            variant.filters_failed.append('q10')
        for callSet in self.getCallSets():
            call = variant.calls.add()
            call.call_set_id = callSet.getId()
            # for now, the genotype is either [0,1], [1,1] or [1,0] with equal
            # probability; probably will want to do something more
            # sophisticated later.
            randomChoice = randomNumberGenerator.choice(
                [[0, 1], [1, 0], [1, 1]])
            call.genotype.extend(randomChoice)
            # TODO What is a reasonable model for generating these likelihoods?
            # Are these log-scaled? Spec does not say.
            call.genotype_likelihood.extend([-100, -100, -100])
        variant.id = self.getVariantId(variant)
        return variant
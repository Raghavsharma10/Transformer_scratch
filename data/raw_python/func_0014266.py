def restraint_force(self, indices=None, strength=5.0):
        """
        Force that restrains atoms to fix their positions, while allowing
        tiny movement to resolve severe clashes and so on.

        Returns
        -------
        force : simtk.openmm.CustomExternalForce
            A custom force to restrain the selected atoms
        """
        if self.system.usesPeriodicBoundaryConditions():
            expression = 'k*periodicdistance(x, y, z, x0, y0, z0)^2'
        else:
            expression = 'k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)'
        force = mm.CustomExternalForce(expression)
        force.addGlobalParameter('k', strength*u.kilocalories_per_mole/u.angstroms**2)
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        positions = self.positions if self.positions is not None else self.handler.positions
        if indices is None:
            indices = range(self.handler.topology.getNumAtoms())
        for i, index in enumerate(indices):
            force.addParticle(i, positions[index].value_in_unit(u.nanometers))
        return force
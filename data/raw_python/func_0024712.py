def _validate_flux_unit(new_unit):  # pragma: no cover
        """Make sure flux unit is valid."""
        new_unit = units.validate_unit(new_unit)

        if new_unit.decompose() != u.dimensionless_unscaled:
            raise exceptions.SynphotError(
                'Unit {0} is not dimensionless'.format(new_unit))

        return new_unit
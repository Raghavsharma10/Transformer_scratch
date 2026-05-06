def modelnumericfunctions(self):
        """Numerical functions of the model class."""
        lines = Lines()
        lines.extend(self.solve)
        lines.extend(self.calculate_single_terms)
        lines.extend(self.calculate_full_terms)
        lines.extend(self.get_point_states)
        lines.extend(self.set_point_states)
        lines.extend(self.set_result_states)
        lines.extend(self.get_sum_fluxes)
        lines.extend(self.set_point_fluxes)
        lines.extend(self.set_result_fluxes)
        lines.extend(self.integrate_fluxes)
        lines.extend(self.reset_sum_fluxes)
        lines.extend(self.addup_fluxes)
        lines.extend(self.calculate_error)
        lines.extend(self.extrapolate_error)
        return lines
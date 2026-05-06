def _mmc_loop(self, rounds, max_angle, max_distance,
                  temp=298.15, stop_when=None, verbose=True):
        """The main Metropolis Monte Carlo loop."""
        current_round = 0
        while current_round < rounds:
            working_model = copy.deepcopy(self.polypeptide)
            random_vector = unit_vector(numpy.random.uniform(-1, 1, size=3))
            mode = random.choice(['rotate', 'rotate', 'rotate', 'translate'])
            if mode == 'rotate':
                random_angle = numpy.random.rand() * max_angle
                working_model.rotate(random_angle, random_vector,
                                     working_model.centre_of_mass)
            else:
                random_translation = random_vector * (numpy.random.rand() *
                                                      max_distance)
                working_model.translate(random_translation)
            proposed_energy = self.eval_fn(working_model, *self.eval_args)
            move_accepted = self.check_move(proposed_energy,
                                            self.current_energy, t=temp)
            if move_accepted:
                self.current_energy = proposed_energy
                if self.current_energy < self.best_energy:
                    self.polypeptide = working_model
                    self.best_energy = copy.deepcopy(self.current_energy)
                    self.best_model = copy.deepcopy(working_model)
            if verbose:
                sys.stdout.write(
                    '\rRound: {}, Current RMSD: {}, Proposed RMSD: {} '
                    '(best {}), {}.       '
                    .format(current_round, self.float_f(self.current_energy),
                            self.float_f(proposed_energy), self.float_f(
                                self.best_energy),
                            "ACCEPTED" if move_accepted else "DECLINED")
                )
                sys.stdout.flush()
            current_round += 1
            if stop_when:
                if self.best_energy <= stop_when:
                    break
        return
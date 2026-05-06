def start_optimisation(self, rounds: int, max_angle: float,
                           max_distance: float, temp: float=298.15,
                           stop_when=None, verbose=None):
        """Starts the loop fitting protocol.

        Parameters
        ----------
        rounds : int
            The number of Monte Carlo moves to be evaluated.
        max_angle : float
            The maximum variation in rotation that can moved per
            step.
        max_distance : float
            The maximum distance the can be moved per step.
        temp : float, optional
            Temperature used during fitting process.
        stop_when : float, optional
            Stops fitting when energy is less than or equal to this value.
        """
        self._generate_initial_score()
        self._mmc_loop(rounds, max_angle, max_distance, temp=temp,
                       stop_when=stop_when, verbose=verbose)
        return
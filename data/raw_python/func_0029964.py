def record_stage_state(self, phase, stage):
        """Record the completion times of phases and stages"""

        key = '{}-{}'.format(phase, stage if stage else 1)

        self.buildstate.state[key] = time()
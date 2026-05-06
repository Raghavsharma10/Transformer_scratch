def pre_gradient_update(self):
        """ First step of Nesterov momentum method:
        take step in direction of accumulated gradient
        """

        updates = zip(self.velocity, self.model.n_parameters * [1.])
        self.model.update_parameters(updates)
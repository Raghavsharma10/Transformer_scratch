def save(self, model_filename, optimizer_filename):
        """ Save the state of the model & optimizer to disk """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(optimizer_filename, self.optimizer)
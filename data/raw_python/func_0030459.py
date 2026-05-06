def get(self, reference, country,  target=datetime.date.today()):
        """
        Get the inflation/deflation value change for the target date based 
        on the reference date. Target defaults to today and the instance's
        reference and country will be used if they are not provided as
        parameters
        """

        # Set country & reference to object's country & reference respectively
        reference = self.reference if reference is None else reference

        # Get the reference and target indices (values) from the source
        reference_value = self.data.get(reference, country).value
        target_value = self.data.get(target, country).value

        # Compute the inflation value and return it
        return self._compute_inflation(target_value, reference_value)
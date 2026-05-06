def reset(self):
        """Clean any processing data, and prepare object for reuse
        """
        self.current_table = None
        self.tables = []
        self.data = [{}]
        self.additional_data = {}
        self.lines = []
        self.set_state('document')
        self.current_file = None
        self.set_of_energies = set()
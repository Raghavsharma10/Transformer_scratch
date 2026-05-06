def run(self):
        """Start the main loop as a background process. *nix only"""
        if win_based:
            raise NotImplementedError("Please run main_loop, "
                                      "backgrounding not supported on Windows")
        self.background_process = mp.Process(target=self.main_loop)
        self.background_process.start()
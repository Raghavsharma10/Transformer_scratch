def graceful_exit(self, msg):
      """ This function Tries to update the MSQL database before exiting. """
      # Print stored errors to stderr
      if self.caught_error:
         self.print2file(self.stderr, False, False, self.caught_error)
      # Kill process with error message
      self.log(msg)
      sys.exit(1)
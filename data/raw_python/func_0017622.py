def output_to_bar(self, message, comma=True):
        """
        Outputs data to stdout, without buffering.

        message: A string containing the data to be output.
        comma: Whether or not a comma should be placed at the end of the output.
        """
        if comma:
            message += ','
        sys.stdout.write(message + '\n')
        sys.stdout.flush()
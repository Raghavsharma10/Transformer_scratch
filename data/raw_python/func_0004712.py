def output(self, message, color=None):
        """
        A helper to used like print() or click's secho() tunneling all the
        outputs to sys.stdout or sys.stderr
        :param message: (str)
        :param color: (str) check click.secho() documentation
        :return: (None) prints to sys.stdout or sys.stderr
        """
        output_to = stderr if color == "red" else stdout
        secho(self.indent(message), fg=color, file=output_to)
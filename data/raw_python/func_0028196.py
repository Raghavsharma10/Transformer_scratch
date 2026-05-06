def _run_morfologik(self, words):
        """
        Runs morfologik java jar and assumes that input and output is
        UTF-8 encoded.
        """
        p = subprocess.Popen(
            ['java', '-jar', self.jar_path, 'plstem',
             '-ie', 'UTF-8',
             '-oe', 'UTF-8'],
            bufsize=-1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        out, _ = p.communicate(input=bytes("\n".join(words), "utf-8"))
        return decode(out, 'utf-8')
def _swarm_breaker(self,
                       seq_path):
        """
            Input : seq_path, a filepath to de-replicated
                    input FASTA reads

            Method: using swarm_breaker.py, break
                    chains of amplicons based on
                    abundance information. Abundance
                    is stored after the final
                    underscore '_' in each sequence
                    label (recommended procedure for
                    Swarm)

            Return: clusters, a list of lists
        """
        swarm_breaker_command = ["swarm_breaker.py",
                                 "-f",
                                 seq_path,
                                 "-s",
                                 self.Parameters['-o'].Value,
                                 "-d",
                                 str(self.Parameters['-d'].Value)]

        try:
            # launch swarm_breaker.py as a subprocess,
            # pipe refined OTU-map to the standard stream
            proc = Popen(swarm_breaker_command,
                         stdout=PIPE,
                         stderr=PIPE,
                         close_fds=True)

            stdout, stderr = proc.communicate()

            if stderr:
                raise StandardError("Process exited with %s" % stderr)

            # store refined clusters in list of lists
            clusters = []
            for line in stdout.split(linesep):
                # skip line if contains only the newline character
                if not line:
                    break
                seq_ids = re.split("\t| ", line.strip())
                # remove the abundance information from the labels
                for i in range(len(seq_ids)):
                    seq_ids[i] = seq_ids[i].rsplit("_", 1)[0]
                clusters.append(seq_ids)
        except OSError:
            raise ApplicationNotFoundError("Cannot find swarm_breaker.py "
                                           "in the $PATH directories.")

        return clusters
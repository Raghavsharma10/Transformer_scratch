def generate_ssh_key(self):
        """
        Generate a new ssh private and public key
        """
        web_command(
            command=["ssh-keygen", "-q", "-t", "rsa", "-N", "", "-C",
                     "datacats generated {0}@{1}".format(
                         getuser(), gethostname()),
                     "-f", "/output/id_rsa"],
            rw={self.profiledir: '/output'},
            )
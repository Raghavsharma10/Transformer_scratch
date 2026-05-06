def _setup_all_outputs(self) -> None:
        """Set up all output methods. Provide them credentials and anything else they need."""

        # The way this is gonna work is that we assume an output should be set up iff it has a
        # credentials_ directory under our secrets dir.
        for key in self.outputs.keys():
            credentials_dir = path.join(self.secrets_dir, f"credentials_{key}")

            # special-case birdsite for historical reasons.
            if key == "birdsite" and not path.isdir(credentials_dir) \
                    and path.isfile(path.join(self.secrets_dir, "CONSUMER_KEY")):
                credentials_dir = self.secrets_dir

            if path.isdir(credentials_dir):
                output_skeleton = self.outputs[key]

                output_skeleton["active"] = True

                obj: Any = output_skeleton["obj"]
                obj.cred_init(secrets_dir=credentials_dir, log=self.log, bot_name=self.bot_name)

                output_skeleton["obj"] = obj

                self.outputs[key] = output_skeleton
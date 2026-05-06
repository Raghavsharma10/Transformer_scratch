def inspect(self, mrf=True):
        """Inspects a Packer Templates file (`packer inspect -machine-readable`)

        To return the output in a readable form, the `-machine-readable` flag
        is appended automatically, afterwhich the output is parsed and returned
        as a dict of the following format:
          "variables": [
            {
              "name": "aws_access_key",
              "value": "{{env `AWS_ACCESS_KEY_ID`}}"
            },
            {
              "name": "aws_secret_key",
              "value": "{{env `AWS_ACCESS_KEY`}}"
            }
          ],
          "provisioners": [
            {
              "type": "shell"
            }
          ],
          "builders": [
            {
              "type": "amazon-ebs",
              "name": "amazon"
            }
          ]

        :param bool mrf: output in machine-readable form.
        """
        self.packer_cmd = self.packer.inspect

        self._add_opt('-machine-readable' if mrf else None)
        self._add_opt(self.packerfile)

        result = self.packer_cmd()
        if mrf:
            result.parsed_output = self._parse_inspection_output(
                                                        result.stdout.decode())
        else:
            result.parsed_output = None
        return result
def _write_branch_and_tag_to_meta_yaml(self):
        """
        Write branch and tag to meta.yaml by editing in place
        """
        ## set the branch to pull source from
        with open(self.meta_yaml.replace("meta", "template"), 'r') as infile:
            dat = infile.read()
            newdat = dat.format(**{'tag': self.tag, 'branch': self.branch})

        with open(self.meta_yaml, 'w') as outfile:
            outfile.write(newdat)
def run_facter(self, key=None):
        """Run the facter executable with an optional specfic
        fact. Output is parsed to yaml if available and
        selected. Puppet facts are always selected. Returns a
        dictionary if no key is given, and the value if a key is
        passed."""
        args = [self.facter_path]
        #this seems to not cause problems, but leaving it separate
        args.append("--puppet")
        if self.external_dir is not None:
            args.append('--external-dir')
            args.append(self.external_dir)
        if self.uses_yaml:
            args.append("--yaml")
        if key is not None:
            args.append(key)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        results = proc.stdout.read()
        if self.uses_yaml:
            parsed_results = yaml.load(results)
            if key is not None:
                return parsed_results[key]
            else:
                return parsed_results
        results = results.decode()
        if key is not None:
            return results.strip()
        else:
            return dict(_parse_cli_facter_results(results))
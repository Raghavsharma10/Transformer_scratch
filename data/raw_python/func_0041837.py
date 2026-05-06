def genchanges(self):
        """Generate a .changes file for this package."""
        chparams = self.params.copy()
        debpath = os.path.join(self.buildroot, self.rule.output_files[0])
        chparams.update({
            'fullversion': '{epoch}:{version}-{release}'.format(**chparams),
            'metahash': self._metahash().hexdigest(),
            'deb_sha1': util.hash_file(debpath, hashlib.sha1()).hexdigest(),
            'deb_sha256': util.hash_file(debpath, hashlib.sha256()
                                         ).hexdigest(),
            'deb_md5': util.hash_file(debpath, hashlib.md5()).hexdigest(),
            'deb_bytes': os.stat(debpath).st_size,
            # TODO: having to do this split('/')[-1] is absurd:
            'deb_filename': debpath.split('/')[-1],
            })

        output = '\n'.join([
            'Format: 1.8',
            # Static date string for repeatable builds:
            'Date: Tue, 01 Jan 2013 00:00:00 -0700',
            'Source: {package_name}',
            'Binary: {package_name}',
            'Architecture: {arch}',
            'Version: {fullversion}',
            'Distribution: {distro}',
            'Urgency: {urgency}',
            'Maintainer: {packager}',
            'Description: ',
            ' {package_name} - {short_description}',
            'Changes: ',
            ' {package_name} ({fullversion}) {distro}; urgency={urgency}',
            ' .',
            ' * Built by Butcher - metahash for this build is {metahash}',
            'Checksums-Sha1: ',
            ' {deb_sha1} {deb_bytes} {deb_filename}',
            'Checksums-Sha256: ',
            ' {deb_sha256} {deb_bytes} {deb_filename}',
            'Files: ',
            ' {deb_md5} {deb_bytes} {section} {priority} {deb_filename}',
            ''  # Newline at end of file.
            ]).format(**chparams)

        return output
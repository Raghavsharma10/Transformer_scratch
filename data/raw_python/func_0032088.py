def manifest(self, subvol):
        """
        Generator for manifest, yields 7-tuples
        """
        subvol_path = os.path.join(self.path, str(subvol))
        builtin_path = os.path.join(subvol_path, MANIFEST_DIR[1:], str(subvol))
        manifest_path = os.path.join(MANIFEST_DIR, str(subvol))

        if os.path.exists(builtin_path):
            # Stream the manifest written into the (read-only) template,
            # note that this has not been done up to now
            return open(builtin_path, "rb")
        elif os.path.exists(manifest_path):
            # Stream the manifest written into /var/lib/butterknife/manifests
            return open(manifest_path, "rb")
        else:
            # If we don't have any stream manifest and save it under /var/lib/butterknife/manifests
            def generator():
                with tempfile.NamedTemporaryFile(prefix=str(subvol), dir=MANIFEST_DIR, delete=False) as fh:
                    print("Temporarily writing to", fh.name)
                    for entry in generate_manifest(os.path.join(self.path, str(subvol))):
                        line = ("\t".join(["-" if j == None else str(j) for j in entry])).encode("utf-8")+b"\n"
                        fh.write(line)
                        yield line
                    print("Renaming to", manifest_path)
                    os.rename(fh.name, manifest_path)
            return generator()
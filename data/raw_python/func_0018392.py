def _downloaded_filename(self):
        """Download the package's archive if necessary, and return its
        filename.

        --no-deps is implied, as we have reimplemented the bits that would
        ordinarily do dependency resolution.

        """
        # Peep doesn't support requirements that don't come down as a single
        # file, because it can't hash them. Thus, it doesn't support editable
        # requirements, because pip itself doesn't support editable
        # requirements except for "local projects or a VCS url". Nor does it
        # support VCS requirements yet, because we haven't yet come up with a
        # portable, deterministic way to hash them. In summary, all we support
        # is == requirements and tarballs/zips/etc.

        # TODO: Stop on reqs that are editable or aren't ==.

        # If the requirement isn't already specified as a URL, get a URL
        # from an index:
        link = self._link() or self._finder.find_requirement(self._req, upgrade=False)

        if link:
            lower_scheme = link.scheme.lower()  # pip lower()s it for some reason.
            if lower_scheme == 'http' or lower_scheme == 'https':
                file_path = self._download(link)
                return basename(file_path)
            elif lower_scheme == 'file':
                # The following is inspired by pip's unpack_file_url():
                link_path = url_to_path(link.url_without_fragment)
                if isdir(link_path):
                    raise UnsupportedRequirementError(
                        "%s: %s is a directory. So that it can compute "
                        "a hash, peep supports only filesystem paths which "
                        "point to files" %
                        (self._req, link.url_without_fragment))
                else:
                    copy(link_path, self._temp_path)
                    return basename(link_path)
            else:
                raise UnsupportedRequirementError(
                    "%s: The download link, %s, would not result in a file "
                    "that can be hashed. Peep supports only == requirements, "
                    "file:// URLs pointing to files (not folders), and "
                    "http:// and https:// URLs pointing to tarballs, zips, "
                    "etc." % (self._req, link.url))
        else:
            raise UnsupportedRequirementError(
                "%s: couldn't determine where to download this requirement from."
                % (self._req,))
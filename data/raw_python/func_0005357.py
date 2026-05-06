def get_filename(self, base_dir=None, modality=None):
        """Construct filename based on the attributes.

        Parameters
        ----------
        base_dir : Path
            path of the root directory. If specified, the return value is a Path,
            with base_dir / sub-XXX / (ses-XXX /) modality / filename
            otherwise the return value is a string.
        modality : str
            overwrite value for modality (i.e. the directory inside subject/session).
            This is necessary because sometimes the modality attribute is ambiguous.

        Returns
        -------
        str or Path
            str of the filename if base_dir is not specified, otherwise the full
            Path
        """
        filename = 'sub-' + self.subject
        if self.session is not None:
            filename += '_ses-' + self.session
        if self.task is not None:
            filename += '_task-' + self.task
        if self.run is not None and self.direction is None:
            filename += '_run-' + self.run
        if self.acquisition is not None:
            filename += '_acq-' + self.acquisition
        if self.direction is not None:
            filename += '_dir-' + self.direction
        if self.run is not None and self.direction is not None:
            filename += '_run-' + self.run
        if self.modality is not None:
            filename += '_' + self.modality
        if self.extension is not None:
            filename += self.extension

        if base_dir is None:
            return filename

        else:
            dir_name = base_dir / ('sub-' + self.subject)
            if self.session is not None:
                dir_name /= 'ses-' + self.session

            if modality is not None:
                dir_name /= modality
            else:
                dir_name = add_modality(dir_name, self.modality)

            return dir_name / filename
def rename_file(self, instance, field_name):
        """
        Renames a file and updates the model field to point to the new file.
        Returns True if a change has been made; otherwise False

        """
        file = getattr(instance, field_name)

        if file:
            new_name = get_hashed_filename(file.name, file)
            if new_name != file.name:
                print('    Renaming "%s" to "%s"' % (file.name, new_name))
                file.save(os.path.basename(new_name), file, save=False)
                return True

        return False
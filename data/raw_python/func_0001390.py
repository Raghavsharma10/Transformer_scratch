def export(self, file_path=None, export_format=None):
        """ Write the users to a file. """
        with io.open(file_path, mode='w', encoding="utf-8") as export_file:
            if export_format == 'yaml':
                import yaml
                yaml.safe_dump(self.to_dict(), export_file, default_flow_style=False)
            elif export_format == 'json':
                export_file.write(text_type(json.dumps(self.to_dict(), ensure_ascii=False)))
            return True
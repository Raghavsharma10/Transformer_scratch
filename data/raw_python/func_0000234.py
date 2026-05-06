def _parse_mat_file(self, file_path):
        """Scan SH12A MAT file for ICRU+LOADEX pairs and return found ICRU numbers"""
        mat_file_sections = self._extract_mat_sections(file_path)
        return self._analyse_mat_sections(mat_file_sections)
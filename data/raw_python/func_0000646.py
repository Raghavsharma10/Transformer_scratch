def open_hierarchy(self, path, relative_to_object_id, object_id, create_file_type=0):
        """
          CreateFileType
          0 - Creates no new object.
          1 - Creates a notebook with the specified name at the specified location.
          2 - Creates a section group with the specified name at the specified location.
          3 - Creates a section with the specified name at the specified location.
        """
        try:
            return(self.process.OpenHierarchy(path, relative_to_object_id, "", create_file_type))
        except Exception as e: 
            print(e)
            print("Could not Open Hierarchy")
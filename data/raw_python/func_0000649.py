def publish(self, hierarchy_id, target_file_path, publish_format, clsid_of_exporter=""):
        """
         PublishFormat
          0 - Published page is in .one format.
          1 - Published page is in .onea format.
          2 - Published page is in .mht format.
          3 - Published page is in .pdf format.
          4 - Published page is in .xps format.
          5 - Published page is in .doc or .docx format.
          6 - Published page is in enhanced metafile (.emf) format.
        """
        try:
            self.process.Publish(hierarchy_id, target_file_path, publish_format, clsid_of_exporter)
        except Exception as e: 
            print(e)
            print("Could not Publish")
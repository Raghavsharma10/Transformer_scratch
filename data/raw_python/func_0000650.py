def get_special_location(self, special_location=0):
        """
          SpecialLocation
          0 - Gets the path to the Backup Folders folder location.
          1 - Gets the path to the Unfiled Notes folder location.
          2 - Gets the path to the Default Notebook folder location.
        """
        try:
            return(self.process.GetSpecialLocation(special_location))
        except Exception as e: 
            print(e)
            print("Could not retreive special location")
def get_asset_path(self, filename):
        """
        Get the full system path of a given asset if it exists.  Otherwise it throws 
        an error.

        Args:
            filename (str) - File name of a file in /assets folder to fetch the path for.

        Returns:
            str - path to the target file.

        Raises:
            AssetNotFoundError - if asset does not exist in the asset folder.

        Usage::
            path = WTF_ASSET_MANAGER.get_asset_path("my_asset.png")
            # path = /your/workspace/location/WTFProjectName/assets/my_asset.png 

        """
        if os.path.exists(os.path.join(self._asset_path, filename)):
            return os.path.join(self._asset_path, filename)
        else:
            raise AssetNotFoundError(
                u("Cannot find asset: {0}").format(filename))
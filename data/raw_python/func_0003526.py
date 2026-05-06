def gdal_nodata_mask(pcl, pcsl, tirs_arr):
        """
        Given a boolean potential cloud layer,
        a potential cloud shadow layer and a thermal band
        Calculate the GDAL-style uint8 mask
        """
        tirs_mask = np.isnan(tirs_arr) | (tirs_arr == 0)
        return ((~(pcl | pcsl | tirs_mask)) * 255).astype('uint8')
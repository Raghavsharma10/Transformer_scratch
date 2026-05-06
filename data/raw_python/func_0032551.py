def _getCampaignDict():
    """Returns a dictionary specifying the details of all campaigns."""
    global _campaign_dict_cache
    if _campaign_dict_cache is None:
        # All pointing parameters and dates are stored in a JSON file
        fn = os.path.join(PACKAGEDIR, "data", "k2-campaign-parameters.json")
        _campaign_dict_cache = json.load(open(fn))
    return _campaign_dict_cache
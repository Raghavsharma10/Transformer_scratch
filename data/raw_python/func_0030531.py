def tzabbr_register(abbr, name, region, zone, dst):
  """Register a new timezone abbreviation in the global registry.

  If another abbreviation with the same name has already been registered it new
  abbreviation will only be registered in region specific dictionary.
  """
  newabbr = tzabbr()
  newabbr.abbr = abbr
  newabbr.name = name
  newabbr.region = region
  newabbr.zone = zone
  newabbr.dst = dst

  if abbr not in all:
    all[abbr] = newabbr

  if not region in regions:
    regions[region] = {}

  assert abbr not in regions[region]
  regions[region][abbr] = newabbr
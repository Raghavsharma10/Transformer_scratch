def adjacent(geohash, direction):
  """Return the adjacent geohash for a given direction."""
  # Based on an MIT licensed implementation by Chris Veness from:
  #   http://www.movable-type.co.uk/scripts/geohash.html
  assert direction in 'nsew', "Invalid direction: %s"%direction
  assert geohash, "Invalid geohash: %s"%geohash
  neighbor = {
    'n': [ 'p0r21436x8zb9dcf5h7kjnmqesgutwvy', 'bc01fg45238967deuvhjyznpkmstqrwx' ],
    's': [ '14365h7k9dcfesgujnmqp0r2twvyx8zb', '238967debc01fg45kmstqrwxuvhjyznp' ],
    'e': [ 'bc01fg45238967deuvhjyznpkmstqrwx', 'p0r21436x8zb9dcf5h7kjnmqesgutwvy' ],
    'w': [ '238967debc01fg45kmstqrwxuvhjyznp', '14365h7k9dcfesgujnmqp0r2twvyx8zb' ]
  }
  border = {
    'n': [ 'prxz',     'bcfguvyz' ],
    's': [ '028b',     '0145hjnp' ],
    'e': [ 'bcfguvyz', 'prxz'     ],
    'w': [ '0145hjnp', '028b'     ]
  }
  last = geohash[-1]
  parent = geohash[0:-1]
  t = len(geohash) % 2
  # Check for edge cases
  if (last in border[direction][t]) and (parent):
    parent = adjacent(parent, direction)
  return parent + BASESEQUENCE[neighbor[direction][t].index(last)]
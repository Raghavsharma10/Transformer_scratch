def get_nickname(userid):
  """Return a Future for a nickname from an account."""
  account = yield get_account(userid)
  if not account:
    nickname = 'Unregistered'
  else:
    nickname = account.nickname or account.email
  raise ndb.Return(nickname)
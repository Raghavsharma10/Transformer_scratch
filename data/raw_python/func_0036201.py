def token_protected_endpoint(function):
  """Requires valid auth_token in POST to access

  An auth_token is built by sending a dictionary built from a
  Werkzeug.Request.form to the scheduler.auth.create_token function.
  """
  @wraps(function)
  def decorated(*args, **kwargs):
    auth_token = request.form.get('auth_token')
    if not auth_token:
      return json.dumps({
        'status': 'fail',
        'reason': 'You must provide an auth_token',
      })

    data = dict(request.form)
    del data['auth_token']
    correct_token = create_token(current_app.config['SECRET_KEY'], data)

    if _compare_digest(auth_token, correct_token):
      return function(*args, **kwargs)

    else:
      return json.dumps({
        'status': 'fail',
        'reason': 'Incorrect auth_token',
      })

  return decorated
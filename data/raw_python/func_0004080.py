def catch_exceptions(orig_func):
    """Catch uncaught exceptions and turn them into http errors"""

    @functools.wraps(orig_func)
    def catch_exceptions_wrapper(self, *args, **kwargs):
        try:
            return orig_func(self, *args, **kwargs)
        except arvados.errors.ApiError as e:
            logging.exception("Failure")
            return {"msg": e._get_reason(), "status_code": e.resp.status}, int(e.resp.status)
        except subprocess.CalledProcessError as e:
            return {"msg": str(e), "status_code": 500}, 500
        except MissingAuthorization:
            return {"msg": "'Authorization' header is missing or empty, expecting Arvados API token", "status_code": 401}, 401
        except ValueError as e:
            return {"msg": str(e), "status_code": 400}, 400
        except Exception as e:
            return {"msg": str(e), "status_code": 500}, 500

    return catch_exceptions_wrapper
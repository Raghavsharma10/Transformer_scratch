def tilecache(self, event, *args, **kwargs):
        """Checks and caches a requested tile to disk, then delivers it to
        client"""
        request, response = event.args[:2]
        self.log(request.path, lvl=verbose)
        try:
            filename, url = self._split_cache_url(request.path, 'tilecache')
        except UrlError:
            return

        # self.log('CACHE QUERY:', filename, url)

        # Do we have the tile already?
        if os.path.isfile(filename):
            self.log("Tile exists in cache", lvl=verbose)
            # Don't set cookies for static content
            response.cookie.clear()
            try:
                yield serve_file(request, response, filename)
            finally:
                event.stop()
        else:
            # We will have to get it first.
            self.log("Tile not cached yet. Tile data: ", filename, url,
                     lvl=verbose)
            if url in self._tiles:
                self.log("Getting a tile for the second time?!", lvl=error)
            else:
                self._tiles += url
            try:
                tile, log = yield self.call(task(get_tile, url), "tcworkers")
                if log:
                    self.log("Thread error: ", log, lvl=error)
            except Exception as e:
                self.log("[MTS]", e, type(e))
                tile = None

            tile_path = os.path.dirname(filename)

            if tile:
                try:
                    os.makedirs(tile_path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        self.log(
                            "Couldn't create path: %s (%s)" % (e, type(e)), lvl=error)

                self.log("Caching tile.", lvl=verbose)
                try:
                    with open(filename, "wb") as tile_file:
                        try:
                            tile_file.write(bytes(tile))
                        except Exception as e:
                            self.log("Writing error: %s" % str([type(e), e]), lvl=error)

                except Exception as e:
                    self.log("Open error on %s - %s" % (filename, str([type(e), e])), lvl=error)
                    return
                finally:
                    event.stop()

                try:
                    self.log("Delivering tile.", lvl=verbose)
                    yield serve_file(request, response, filename)
                except Exception as e:
                    self.log("Couldn't deliver tile: ", e, lvl=error)
                    event.stop()
                self.log("Tile stored and delivered.", lvl=verbose)
            else:
                self.log("Got no tile, serving default tile: %s" % url)
                if self.default_tile:
                    try:
                        yield serve_file(request, response, self.default_tile)
                    except Exception as e:
                        self.log('Cannot deliver default tile:', e, type(e),
                                 exc=True, lvl=error)
                    finally:
                        event.stop()
                else:
                    yield
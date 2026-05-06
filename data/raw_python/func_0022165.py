def init_jcrop(min_size=None):
        """Initialize jcrop.

        :param min_size: The minimal size of crop area.
        """
        init_x = current_app.config['AVATARS_CROP_INIT_POS'][0]
        init_y = current_app.config['AVATARS_CROP_INIT_POS'][1]
        init_size = current_app.config['AVATARS_CROP_INIT_SIZE'] or current_app.config['AVATARS_SIZE_TUPLE'][2]

        if current_app.config['AVATARS_CROP_MIN_SIZE']:
            min_size = min_size or current_app.config['AVATARS_SIZE_TUPLE'][2]
            min_size_js = 'jcrop_api.setOptions({minSize: [%d, %d]});' % (min_size, min_size)
        else:
            min_size_js = ''
        return Markup('''
<script type="text/javascript">
    jQuery(function ($) {
      // Create variables (in this scope) to hold the API and image size
      var jcrop_api,
          boundx,
          boundy,

          // Grab some information about the preview pane
          $preview = $('#preview-box'),
          $pcnt = $('#preview-box .preview-box'),
          $pimg = $('#preview-box .preview-box img'),

          xsize = $pcnt.width(),
          ysize = $pcnt.height();

      $('#crop-box').Jcrop({
        onChange: updatePreview,
        onSelect: updateCoords,
        setSelect: [%s, %s, %s, %s],
        aspectRatio: 1
      }, function () {
        // Use the API to get the real image size
        var bounds = this.getBounds();
        boundx = bounds[0];
        boundy = bounds[1];
        // Store the API in the jcrop_api variable
        jcrop_api = this;
        %s
        jcrop_api.focus();
        // Move the preview into the jcrop container for css positioning
        $preview.appendTo(jcrop_api.ui.holder);
      });

      function updatePreview(c) {
        if (parseInt(c.w) > 0) {
          var rx = xsize / c.w;
          var ry = ysize / c.h;
          $pimg.css({
            width: Math.round(rx * boundx) + 'px',
            height: Math.round(ry * boundy) + 'px',
            marginLeft: '-' + Math.round(rx * c.x) + 'px',
            marginTop: '-' + Math.round(ry * c.y) + 'px'
          });
        }
      }
    });

    function updateCoords(c) {
      $('#x').val(c.x);
      $('#y').val(c.y);
      $('#w').val(c.w);
      $('#h').val(c.h);
    }
  </script>
            ''' % (init_x, init_y, init_size, init_size, min_size_js))
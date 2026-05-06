def _aggregate(self):
        '''
        The function aggreagtes all pre_stream, layout and post_stream and
        components, and yields them one by one.
        '''
        # Yielding everything under pre_stream
        for x in self._yield_all(self.pre_stream): yield x

        # Yielding layout
        for x in self._yield_all(self.layout): yield x

        # Yield LayoutSR Specific Content
        yield """
        <div id="%s">
        <script>
            var MyFlaskSRMutationObserver = (function () {
                var prefixes = ['WebKit', 'Moz', 'O', 'Ms', '']
                for(var i=0; i < prefixes.length; i++) {
                    if(prefixes[i] + 'MutationObserver' in window) {
                        return window[prefixes[i] + 'MutationObserver'];
                    }
                }
                return false;
            }());

            if(MyFlaskSRMutationObserver) {
                var target = document.getElementById('%s');
                var observerFlaskSR = new MyFlaskSRMutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        var obj = mutation.addedNodes[0];
                        if(obj instanceof HTMLElement) {
                            var referenceId = obj.getAttribute('sr-id');
                            if(referenceId) {
                                document.getElementById('%s').querySelectorAll("*[ref-sr-id='"+referenceId+"']")[0].innerHTML = obj.innerHTML;
                                obj.innerHTML = '';
                            }
                        }
                    });
                });
                var config = {
                    childList: true
                }
                observerFlaskSR.observe(target, config);
                document.addEventListener("DOMContentLoaded", function(event) {
                    observerFlaskSR.disconnect();
                });
            }
            else {
                console.log("MutationObserver not found!");
            }
        </script>""" % (
            self.stream_div_id,
            self.stream_div_id,
            self.stream_div_layout_id
        )

        # Yielding components
        for x in self._yield_all(self.components): yield x

        # Yield LayoutSR Specific Content
        yield """</div>"""

        # Yielding everything under post_stream
        for x in self._yield_all(self.post_stream): yield x
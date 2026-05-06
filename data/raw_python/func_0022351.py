def code_toggle(start_show=False, message=None):
    """Toggling on and off code in a notebook. 
    :param start_show: Whether to display the code or not on first load (default is False).
    :type start_show: bool
    :param message: the message used to toggle display of the code.
    :type message: string

    The tip that this idea is
    based on is from Damian Kao (http://blog.nextgenetics.net/?e=102)."""

    
    html ='<script>\n'
    if message is None:
        message = u'The raw code for this jupyter notebook can be hidden for easier reading.'
    if start_show:
        html += u'code_show=true;\n'
    else:
        html += u'code_show=false;\n'
    html+='''function code_toggle() {
 if (code_show){
 $('div.input').show();
 } else {
 $('div.input').hide();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
'''
    html += message + ' To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.' 
    display(HTML(html))
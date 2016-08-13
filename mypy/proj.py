import os
import json

# add Project class - can read yaml project setup info
#                   - governs path etc.
# basic project structure:
#   code
#   notebooks
#   data
#   fig
#

def find_dropbox():
    app = os.getenv('APPDATA')
    drp_pth = app[:3] + os.path.join(*app[3:].split('\\')[:-1],
                                     'Local', 'Dropbox')
    if os.path.exists(drp_pth):
        info_file = os.path.join(drp_pth, 'info.json')
        with open(info_file) as f:
            info = json.load(f)
        return info['personal']['path']

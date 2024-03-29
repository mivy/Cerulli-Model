# possibly the only way to get modules (i.e. pandas) on bpy for a headless server
#
# C:\> blender -b "default.blend" -P install_pandas.py
#
#

import subprocess
import sys
import os
 
# path to python.exe
python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
 
# upgrade pip
subprocess.call([python_exe, "-m", "ensurepip"])
subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
 
# install required packages
subprocess.call([python_exe, "-m", "pip", "install", "pandas"])

#
#
# code from here:
# https://b3d.interplanety.org/en/how-to-install-required-packages-to-the-blender-python-with-pip/

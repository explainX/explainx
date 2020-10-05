import os
from sys import platform


if platform == "linux" or platform == "linux2":
    try:
        os.system("sudo apt install nodejs")
        os.system("sudo apt install npm")
    except:
        os.system("sudo yum install nodejs")
        os.system("sudo yum install npm")
    os.system("npm install -g localtunnel")

elif platform == "darwin":
    os.system("xcode-select --install")
    os.system("brew install nodejs")
    os.system("npm install -g localtunnel")

elif platform == "win32":
    print("Please install nodejs, npm, and localtunnel manually")
    os.system("npm install -g localtunnel")
elif platform == "win64":
    print("Please install nodejs, npm, and localtunnel manually")
    os.system("npm install -g localtunnel")


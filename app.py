import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

exec(open("src/main.py").read())
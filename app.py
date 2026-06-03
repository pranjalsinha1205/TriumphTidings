import runpy
import sys

sys.argv = ["streamlit", "run", "src/main.py", "--server.port=7860", "--server.address=0.0.0.0"]
runpy.run_module("streamlit", run_name="__main__")
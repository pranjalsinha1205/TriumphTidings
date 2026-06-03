from streamlit.web import cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/main.py"]
    sys.exit(stcli.main())
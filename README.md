# Neurvis
## Virtual Environment
### To make new virtual environment
- make sure u have python 3.12
mac:
brew install python@3.12

windows:
winget install Python.Python.3.12

Opt 1 - enter in terminal:
py -3.12 -m venv .venv

Opt 2 - Use VSC Command Palette
select 'Python: Select Interpreter' then pick the 3.12 one
make sure default interpreter in vsc settings (command or control + ,) is 3.12 python path which u can find by:
mac:
python3.12 -c "import sys; print(sys.executable)"

windows:
py -3.12 -c "import sys; print(sys.executable)"


### To activate current virtual environment
1. Make sure the virtual environment exist
2. Enter in terminal:
mac:
source .venv/bin/activate

windows:
.\.venv\Scripts\activate

### Don't forget to install dependencies after creation and activation of virtual environment
In terminal. run:
pip install -r requirements.txt

### Interpreter Selection
Make sure in vsc command palette set Python Interpreter to the one with 'workspace' tag beside environment name/path


import os
import subprocess
import shutil

# Delete db folder if it already exists then create it again
db_folder = os.path.join(os.getcwd(), 'db')
if os.path.exists(db_folder):
    shutil.rmtree(db_folder)
os.makedirs(db_folder)

# List of scripts to run
scripts = [
    'src/1-ETL/dataPreparationAPI.py',
    'src/1-ETL/dataPreparationBao.py',
    'src/1-ETL/dataPreparationPubChem.py'
]

# Run each script
for script in scripts:
    script_path = os.path.join(os.getcwd(), script)
    if os.path.exists(script_path):
        subprocess.run(['python', script_path])
    else:
        print(f"Script {script} not found.")
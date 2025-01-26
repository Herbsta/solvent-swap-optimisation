import os
import subprocess
import shutil

from dataPreparationPubChem import main as mainPubChem
from dataPreparationAPI import main as mainAPI
from dataPreparationBao import main as mainBao

from getMWfromCompoundUsingPubChem import main as getMWfromCompoundUsingPubChem
from getMWFromCIDPubChem import main as getMWFromCIDPubChem


# Delete db folder if it already exists then create it again
db_folder = os.path.join(os.getcwd(), 'db')
if os.path.exists(db_folder):
    shutil.rmtree(db_folder)
os.makedirs(db_folder)

# Run the ETL process for each data source
mainPubChem()
mainAPI()
mainBao()

# Run the fetching of MW for each compiund

getMWfromCompoundUsingPubChem()
getMWFromCIDPubChem()

# Convert the database gram per gram to mol per mol
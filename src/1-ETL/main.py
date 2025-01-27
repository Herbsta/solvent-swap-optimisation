import os
import shutil

from dataPreparationPubChem import main as mainPubChem
from dataPreparationAPI import main as mainAPI
from dataPreparationBao import main as mainBao
from dbCombiner import main as dbCombiner

# Delete db folder if it already exists then create it again
db_folder = os.path.join(os.getcwd(), 'db')
if os.path.exists(db_folder):
    shutil.rmtree(db_folder)
os.makedirs(db_folder)

mainPubChem()
mainAPI()
mainBao()

# Combine the databases
dbCombiner()
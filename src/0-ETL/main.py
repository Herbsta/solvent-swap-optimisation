import os
import shutil

from dataPreparationPubChem import main as mainPubChem
from dataPreparationAPI import main as mainAPI
from dataPreparationBao import main as mainBao
from dbCombiner import main as dbCombiner
from getPropertiesPubChem import main as getPropertiesPubChem
from dataRedundancy import main as dataRedundancy
from dataBaoCombiner1 import main as Bao1
from dataBaoCombiner2 import main as Bao2
from feature_combiner import main as feature_combiner

# Delete db folder if it already exists then create it again
db_folder = os.path.join(os.getcwd(), 'db')
if os.path.exists(db_folder):
    shutil.rmtree(db_folder)
os.makedirs(db_folder)

# Run the ETL process for each data source
mainPubChem()
mainAPI()
mainBao()

# Combine the databases
dbCombiner()


# Get properties from PubChem
getPropertiesPubChem()

# Add Bao et al. data to the mix
Bao1()
Bao2()

# Remove all unnecessary data values & conversion
dataRedundancy()

# Combine the features
feature_combiner()



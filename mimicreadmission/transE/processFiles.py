import pandas as pd
from tqdm import tqdm

# List of filenames for each relation type
filenames = [
    "AnnotationsFinalDiagnosis.csv",
    "AnnotationsInitialDiagnosis.csv"
]

# Initialize lists to hold the data for all files
allData = []
allRelations = set() 
#print(all_relations) # To hold all unique relation types
allEntities = set()  # To hold all unique entities (Entity1 and Entity2)

# Read and process each file
for filename in filenames:
    if 'Prescriptions' in filename:
        data = pd.read_csv(f'/Users/ricardocarvalho/Documents/Thesis/OutFiles/{filename}', sep=';', header=None, names=["Entity1", "Entity2", "StartTime", "EndTime"])
    else:
        data = pd.read_csv(f'/Users/ricardocarvalho/Documents/Thesis/OutFiles/{filename}', sep=';',header=None, names=["Entity1", "Entity2","Time"])
    
    # Add data to the all_data list
    allData.append(data)

    relationType = f'has{filename[11:-4]}'  # Get relation type from filename
    allRelations.add(relationType)

    # Add Entity1 and Entity2 to the entities set
    allEntities.update(data["Entity1"].unique())
    allEntities.update(data["Entity2"].unique())

# Create the entity2id mapping (Entity -> ID)
entity2idMapping = {entity: idx for idx, entity in enumerate(sorted(allEntities))}

# Create the relation2id mapping (Relation -> ID)
relation2idMapping = {relation: idx for idx, relation in enumerate(sorted(allRelations))}

# Write the entity2id file
with open("mimicreadmission/transE/outFiles/entity2id.txt", "w") as f:
    for entity, idx in entity2idMapping.items():
        f.write(f"{idx}\t{entity}\n")

# Write the relation2id file
with open("mimicreadmission/transE/outFiles/relation2id.txt", "w") as f:
    for relation, idx in relation2idMapping.items():
        f.write(f"{idx}\t{relation}\n")

# Generate the train2id file
# Open the output file to save triples
with open("mimicreadmission/transE/outFiles/train2id.txt", "w") as f:
    # Iterate over each DataFrame in allData
    for data_index, data in enumerate(tqdm(allData, desc="Processing DataFrames")):
        # Check the DataFrame's data types (for debugging)
        print(data.dtypes)
        
        # Generate the relation type from the filename
        relationType = f'has{filenames[data_index][11:-4]}'
        relationid = relation2idMapping[relationType]

        # Iterate over each row in the DataFrame
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Rows", leave=False):
            # Map Entity1 and Entity2 to their corresponding IDs
            entity1id = entity2idMapping.get(row["Entity1"])
            entity2id = entity2idMapping.get(row["Entity2"])
            
            # Write the triple to the train2id file
            f.write(f"{entity1id}\t{relationid}\t{entity2id}\n")

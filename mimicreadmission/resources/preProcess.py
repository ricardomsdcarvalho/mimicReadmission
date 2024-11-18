import rdflib
from rdflib import URIRef
import buildGraph as BG
import tqdm

def process_kg_for_transe(kg):
    entities = set()
    relations = set()
    triples = []

    for s, p, o in tqdm(kg, desc="Processing Triples", unit="triple"):
        # Add subject, predicate, and object URIs to entities and relations sets
        entities.add(str(s))
        relations.add(str(p))
        entities.add(str(o))
        triples.append((str(s), str(p), str(o)))
    
    return list(entities), list(relations), triples

def save_mappings_and_triples(kg, output_path = 'mimicreadmission/data/entityRelation' ):

    entities, relations, triples = process_kg_for_transe(kg)
    # Create entity and relation dictionaries with unique IDs
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    # Save entity2id.txt
    with open(output_path + 'entity2id.txt', 'w') as f:
        f.write(f"{len(entity2id)}\n")  # Number of entities
        for entity, idx in tqdm(entity2id.items(), desc="Writing Entities", unit="entity"):
            f.write(f"{entity}\t{idx}\n")

    # Save relation2id.txt
    with open(output_path + 'relation2id.txt', 'w') as f:
        f.write(f"{len(relation2id)}\n")  # Number of relations
        for relation, idx in tqdm(relation2id.items(), desc="Writing Relations", unit="relation"):
            f.write(f"{relation}\t{idx}\n")

    # Save train2id.txt
    with open(output_path + 'train2id.txt', 'w') as f:
        f.write(f"{len(triples)}\n")  # Number of triples
        for (head, rel, tail) in tqdm(triples, desc="Writing Triples", unit="triple"):
            f.write(f"{entity2id[head]}\t{entity2id[tail]}\t{relation2id[rel]}\n")

if __name__ == '__main__':
    # Load the knowledge graph
    ontologySet = ['mimicreadmission/data/ontologies/Thesaurus.owl',
                   'mimicreadmission/data/ontologies/LOINC.rdf',
                   'mimicreadmission/data/ontologies/dron.owl',
                    'mimicreadmission/data/ontologies/ICD9CM.ttl'
                   ]
    annotationSet = ['mimicreadmission/data/annotations/AnnotationsInitialDiagnosis.csv',
                    'mimicreadmission/data/annotations/AnnotationsLabEvents.csv',
                    'mimicreadmission/data/annotations/AnnotationsPrescriptions.csv',
                    'mimicreadmission/data/annotations/AnnotationsProcedures.csv',
                    'mimicreadmission/data/annotations/AnnotationsFinalDiagnosis.csv'
                    ]
                    
    annotationType = ['hasInitialDiagnosis',
                    'hasLabEvent',
                    'hasPrescription',
                    'hasProcedure',
                    'hasFinalDiagnosis'
                      ]
    
    kg, entities = BG.construct_kg(ontologySet, annotationSet, annotationType)
    
    # Save entity2id.txt, relation2id.txt, and train2id.txt
    save_mappings_and_triples(kg)
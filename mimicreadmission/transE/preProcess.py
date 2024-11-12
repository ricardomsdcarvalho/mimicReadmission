import rdflib
from rdflib import URIRef

def process_kg_for_transe(kg):
    entities = set()
    relations = set()
    triples = []

    for s, p, o in kg:
        # Add subject, predicate, and object URIs to entities and relations sets
        entities.add(str(s))
        relations.add(str(p))
        entities.add(str(o))
        triples.append((str(s), str(p), str(o)))
    
    return list(entities), list(relations), triples

def save_mappings_and_triples(kg, output_path = 'mimicreadmission/transE/files/' ):

    entities, relations, triples = process_kg_for_transe(kg)
    # Create entity and relation dictionaries with unique IDs
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    # Save entity2id.txt
    with open(output_path + 'entity2id.txt', 'w') as f:
        f.write(f"{len(entity2id)}\n")  # Number of entities
        for entity, idx in entity2id.items():
            f.write(f"{entity}\t{idx}\n")

    # Save relation2id.txt
    with open(output_path + 'relation2id.txt', 'w') as f:
        f.write(f"{len(relation2id)}\n")  # Number of relations
        for relation, idx in relation2id.items():
            f.write(f"{relation}\t{idx}\n")

    # Save train2id.txt
    with open(output_path + 'train2id.txt', 'w') as f:
        f.write(f"{len(triples)}\n")  # Number of triples
        for (head, rel, tail) in triples:
            f.write(f"{entity2id[head]}\t{entity2id[tail]}\t{relation2id[rel]}\n")

from operator import itemgetter

import rdflib
from rdflib.namespace import RDF, OWL, RDFS
from rdflib import URIRef

from pyrdf2vec.graphs import kg
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.samplers import UniformSampler, PredFreqSampler
from pyrdf2vec.walkers import WeisfeilerLehmanWalker



# The format of the input files is allwaays: (id);[annoations]
# as such we can always split by ; and than loop the list 

def construct_kg(ontologySet,annotationSet,annotationType):
    try:
        assert len(ontologySet) == len(annotationSet), "The number of ontologies and annotations must be the same"
        # Proceed with the rest of the function if assertion passes
        print("Ontologies and annotations are correctly matched.")

        kg = rdflib.Graph()
        ents = set()

        for loc in range(len(ontologySet)):
            kg.parse(ontologySet[loc], format='xml')

            fileAnnotations = open(annotationSet[loc], 'r')
            for annot in fileAnnotations.readlines()[:]:
                annot = annot.lstrip()

                #Get Head and Tail Entities
                headInfo, annotations = annot.split(';')
                headEnt = f'http://purl.obolibrary.org/obo/{headInfo.split(',')[0]}'

                ents.update((ent.strip() for ent in annotations.split(',')))

                for urlAnnot in annotations.split(','):
                    kg.add((URIRef(headEnt), URIRef(f'http://purl.obolibrary.org/obo/{annotationType[loc]}'),
                            URIRef(urlAnnot.strip())))
            
            fileAnnotations.close()
            kg.add(URIRef(f'http://purl.obolibrary.org/obo/{annotationType[loc]}'), RDF.type, OWL.ObjectProperty)

    except AssertionError as error:
        print(f"Error: {error} Because the number of ontologies, annotations must be the same")

    #kg.serialize(destination='mimicreadmission/rdf2vec/outFiles/myKG.xml')

    print('KG created')
    return kg, list(ents)



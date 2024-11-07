import numpy
import os
from operator import itemgetter
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
from rdflib import URIRef
import json

import buildGraph as BG
from pyrdf2vec.graphs import kg
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec

from pyrdf2vec.samplers import UniformSampler, PredFreqSampler
from pyrdf2vec.samplers.frequency import ObjFreqSampler, ObjPredFreqSampler
from pyrdf2vec.samplers.pagerank import PageRankSampler

from pyrdf2vec.walkers import WeisfeilerLehmanWalker
from pyrdf2vec.walkers.anonymous import AnonymousWalker
from pyrdf2vec.walkers.halk import HalkWalker
from pyrdf2vec.walkers.ngrams import NGramWalker
from pyrdf2vec.walkers.random import RandomWalker
from pyrdf2vec.walkers.walklets import WalkletWalker


def calculate_embeddings(knowledgeGraph, entities, outputPath = 'mimicreadmission/rdf2vec/outFiles', 
                        size_value = 300, type_word2vec = 'skip-gram', n_walks = 500, walk_depth = 4, 
                        walker_type = 'wl' ,sampler_type = 'uniform'):
    
    graph = kg.rdflib_to_kg(knowledgeGraph)

    ######### Word2Vec type #########
    if type_word2vec == 'CBOW':
        sg_value = 0
    elif type_word2vec == 'skip-gram':
        sg_value = 1
    
    ######### Sampler type #########
    if sampler_type.lower() == 'uniform':
        sampler = UniformSampler()
    elif sampler_type.lower() == 'predfreq':
        sampler = PredFreqSampler()
    elif sampler_type.lower() == 'objfreq':
        sampler = ObjFreqSampler()
    elif sampler_type.lower() == 'objpredfreq':
        sampler = ObjPredFreqSampler()
    elif sampler_type.lower() == 'pagerank':
        sampler = PageRankSampler()
    
    ######### Walker type #########
    if walker_type.lower() == 'random':
        walker = RandomWalker(depth=walk_depth, walks_per_graph=n_walks, sampler=sampler)
    elif walker_type.lower() == 'wl':
        walker = WeisfeilerLehmanWalker(depth=walk_depth, walks_per_graph=n_walks, sampler=sampler)
    elif walker_type.lower() == 'anonymous':
        walker = AnonymousWalker(depth=walk_depth, walks_per_graph=n_walks, sampler=sampler)
    elif walker_type.lower() == 'halk':
        walker = HalkWalker(depth=walk_depth, walks_per_graph=n_walks, sampler=sampler)
    elif walker_type.lower() == 'ngram':
        walker = NGramWalker(depth=walk_depth, walks_per_graph=n_walks, sampler=sampler)
    elif walker_type.lower() == 'walklet':
        walker = WalkletWalker(depth=walk_depth, walks_per_graph=n_walks, sampler=sampler)

    ######### Print the parameters #########
    print(f'{10*'#'}\n Vector size: {size_value}\n')
    print(f'Type Word2vec: {type_word2vec}\n')
    print(f'Type Walker: {walker_type}\n{10*'#'}\n')

    ######### Create the transformer #########
    transformer = RDF2VecTransformer(Word2Vec(size=size_value, sg=sg_value), walkers=[walker])
    print('Transformer run')
    embeddings = transformer.fit_transform(graph, entities)

    ######### Write the files #########
    with open(outputPath + f'Embeddings_{type_word2vec}_{size_value}.txt', 'w') as file:
        for entity, embedding in zip(entities, embeddings):
            # Join the entity and its embedding values as a single line
            line = f"{entity} " + " ".join(map(str, embedding))
            file.write(line + '\n')
    
    print('Embeddings saved in ' + outputPath + f'Embeddings_{type_word2vec}_{size_value}.txt')


def main():
    # Ontologies and annotations
    ontologySet = ['mimicreadmission/rdf2vec/inFiles/ontology1.xml', 'mimicreadmission/rdf2vec/inFiles/ontology2.xml']
    annotationSet = ['mimicreadmission/rdf2vec/inFiles/annotation1.txt', 'mimicreadmission/rdf2vec/inFiles/annotation2.txt']
    annotationType = ['hasAnnotation1', 'hasAnnotation2']

    # Construct the Knowledge Graph
    Kg, ents = BG.construct_kg(ontologySet, annotationSet, annotationType)

    # Calculate the embeddings
    calculate_embeddings(Kg, ents)


if __name__ == '__main__':
    main()
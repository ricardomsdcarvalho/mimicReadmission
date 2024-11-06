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
from pyrdf2vec.walkers import WeisfeilerLehmanWalker


def calculateEmbeddings(KnowledgeGraph, entities, outputPath, sizeValue, windowValue, minCountValue, workersValue, iterValue):
    
    return embeddings
    
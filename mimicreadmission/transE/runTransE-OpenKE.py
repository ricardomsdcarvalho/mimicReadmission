from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
import preProcess 
import mimicreadmission.buildGraph as BG
import numpy as np


def main(ontologySet, annotationSet, annotationType, inpath = 'mimicreadmission/transE/outFiles/'):
    # Make inputFiles for embedding process
    knowledgeGraph = BG.construct_kg(ontologySet, annotationSet, annotationType)
    preProcess.save_mappings_and_triples(knowledgeGraph)

    # Load training data
    train_dataloader = TrainDataLoader(
        in_path = inpath,  # Replace with the path where your files are located
        nbatches = 100,       # Number of batches
        threads = 8,          # Number of threads for multi-threaded data loading
        sampling_mode = "normal",
        bern_flag = 0,
        filter_flag = 1,
        neg_ent = 25,
        neg_rel = 0
    )

    # Set up TransE model
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 300,             # Embedding dimension
        p_norm = 1,
        norm_flag = True
    )
   
    # Define loss function
    model = NegativeSampling(
        model = transe,
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size()
    )

    # Train the model
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
    trainer.run()
    transe.save_checkpoint("./checkpoint/transe.ckpt")

    # Load trained TransE model and export embeddings
    transe.load_checkpoint("./checkpoint/transe.ckpt")

    # Get entity embeddings
    ent_embeddings = transe.ent_embeddings.weight.cpu().data.numpy()
    rel_embeddings = transe.rel_embeddings.weight.cpu().data.numpy()

    # Save entity embeddings
    np.savetxt('./output/entity_embeddings.txt', ent_embeddings, delimiter='\t')
    np.savetxt('./output/relation_embeddings.txt', rel_embeddings, delimiter='\t')


if __name__ == '__main__':
    # Ontologies and annotations
    ontologySet = ['mimicreadmission/rdf2vec/inFiles/ontology1.xml', 'mimicreadmission/rdf2vec/inFiles/ontology2.xml']
    annotationSet = ['mimicreadmission/rdf2vec/inFiles/annotation1.txt', 'mimicreadmission/rdf2vec/inFiles/annotation2.txt']
    annotationType = ['hasAnnotation1', 'hasAnnotation2']

    main(ontologySet, annotationSet, annotationType)
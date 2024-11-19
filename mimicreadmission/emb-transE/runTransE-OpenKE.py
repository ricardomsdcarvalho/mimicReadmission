import sys

sys.path.append('/Users/ricardocarvalho/Documents/WorkStation/mimicReadmission/mimicreadmission/resources/OpenKE')

from openke.config import Trainer
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
import numpy as np
import os


def main(inpath = '/Users/ricardocarvalho/Documents/WorkStation/mimicReadmission/mimicreadmission/data/entityRelation/'):
    
    # Define the expected file paths in the output directory
    output_files = [
        f'{inpath}entity2id.txt',
        f'{inpath}relation2id.txt',
        f'{inpath}train2id.txt'
    ]

    # Check if all expected files exist
    assert all(os.path.exists(file) for file in output_files), (
        "Required output files are missing. Please run the script mimicreadmission/preProcess.py \
            and ensure the following files are in the output directory:\n" +
        "\n".join(output_files)
    )

    print("All output files are present. Proceeding with the next steps.")

    # Load training data
    train_dataloader = TrainDataLoader(
        in_path = inpath,  
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
    trainer = Trainer(model = model, 
                      data_loader = train_dataloader, 
                      train_times = 1000, 
                      alpha = 0.5, 
                      use_gpu = True)
    trainer.run()
    transe.save_checkpoint("./checkpoint/transe.ckpt")

    # Load trained TransE model and export embeddings
    transe.load_checkpoint("./checkpoint/transe.ckpt")

    # Get entity embeddings
    ent_embeddings = transe.ent_embeddings.weight.cpu().data.numpy()
    rel_embeddings = transe.rel_embeddings.weight.cpu().data.numpy()

    # Save entity embeddings
    np.savetxt('/Users/ricardocarvalho/Documents/WorkStation/mimicReadmission/mimicreadmission/emb-transE/output/entity_embeddings.txt', ent_embeddings, delimiter='\t')
    np.savetxt('/Users/ricardocarvalho/Documents/WorkStation/mimicReadmission/mimicreadmission/emb-transE/output/relation_embeddings.txt', rel_embeddings, delimiter='\t')


if __name__ == '__main__':
    
    main()
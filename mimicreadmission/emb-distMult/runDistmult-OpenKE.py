import sys
sys.path.append('/home/ricciard0.dc/mimicReadmission/mimicreadmission/resources/OpenKE')

from openke.config import Trainer
from openke.module.model import DistMult
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
import numpy as np
import os

def main(inpath='/home/ricciard0.dc/mimicReadmission/mimicreadmission/Data/entityRelation/'):
    
    # Define the expected file paths in the output directory
    output_files = [
        f'{inpath}entity2id.txt',
        f'{inpath}relation2id.txt',
        f'{inpath}train2id.txt'
    ]

    # Use assert to check if all expected files exist
    assert all(os.path.exists(file) for file in output_files), (
       "Required output files are missing. Please run the script mimicreadmission/preProcess.py \
        and ensure the following files are in the output directory:\n" +
       "\n".join(output_files)
    )

    print("All output files are present. Proceeding with the next steps.")

    # Load training data
    train_dataloader = TrainDataLoader(
        in_path=inpath, 
        nbatches=500,       # Number of batches
        threads=8,          # Number of threads
        sampling_mode="normal",
        bern_flag=0,
        filter_flag=1,
        neg_ent=10,         # Number of negative samples for entities
        neg_rel=0           # Number of negative samples for relations
    )

    # Set up DistMult model
    distmult = DistMult(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=128             # Embedding dimension
    )

    # Define pairwise margin loss
    model = NegativeSampling(
        model=distmult,
        loss=MarginLoss(margin=5.0),  # Pairwise margin loss with a margin of 5.0
        batch_size=train_dataloader.get_batch_size()
    )

    # Train the model
    trainer = Trainer(
        model=model, 
        data_loader=train_dataloader, 
        train_times=500,     # Number of epochs
        alpha=0.5,           # Learning rate
        use_gpu=True
    )
    
    trainer.run()
    distmult.save_checkpoint("/home/ricciard0.dc/mimicReadmission/mimicreadmission/emb-distMult/checkpoint/distmult.ckpt")

    # Load the trained DistMult model and export the embeddings
    distmult.load_checkpoint("/home/ricciard0.dc/mimicReadmission/mimicreadmission/emb-distMult/checkpoint/distmult.ckpt")

    # Get the entity embeddings
    ent_embeddings = distmult.ent_embeddings.weight.data.cpu().data.numpy()
    rel_embeddings = distmult.rel_embeddings.weight.data.cpu().data.numpy()

    # Save entity embeddings
    np.savetxt('/home/ricciard0.dc/mimicReadmission/mimicreadmission/emb-distMult/output/distMult_128_Eembeddings.txt', ent_embeddings, delimiter='\t')
    np.savetxt('/home/ricciard0.dc/mimicReadmission/mimicreadmission/emb-distMult/output/distMult_128_Rembeddings.txt', rel_embeddings, delimiter='\t')

if __name__ == '__main__':
    main()

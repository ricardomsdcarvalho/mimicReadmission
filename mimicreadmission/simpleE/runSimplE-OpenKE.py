from openke.config import Trainer
from openke.module.model import SimpleE
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
import numpy as np
import os


def main(inpath = 'mimicreadmission/Data/entityRelation'):
    
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
        in_path = inpath, 
        nbatches = 100,       # Number of batches
        threads = 8,          # Number of threads
        sampling_mode = "normal",
        bern_flag = 0,
        filter_flag = 1,
        neg_ent = 25,
        neg_rel = 0
    )

    # Set up SimpleE model
    simplee = SimpleE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 300             # Embedding dimension
    )

    # Define loss function for SimpleE
    model = NegativeSampling(
        model = simplee,
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size()
    )

    # Train the model
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
    trainer.run()
    simplee.save_checkpoint("./checkpoint/simplee.ckpt")

    #Load the trained simplee model and export the embeddings
    simplee.load_checkpoint("./checkpoint/simplee.ckpt")

    # Get the entity embeddings
    ent_embeddings = simplee.ent_embeddings.weight.data.cpu().data.numpy()
    rel_embeddings = simplee.rel_embeddings.weight.data.cpu().data.numpy()

    # Save entity embeddings
    np.savetxt('./output/entity_embeddings.txt', ent_embeddings, delimiter='\t')
    np.savetxt('./output/relation_embeddings.txt', rel_embeddings, delimiter='\t')

if __name__ == '__main__':
    
    main()
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
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
        nbatches = 100,
        threads = 8,
        sampling_mode = "normal",
        bern_flag = 1,
        filter_flag = 1,
        ng_ents = 25,
        ng_rels = 0
    )

    # set up complEx model
    complEx = ComplEx(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 300
    )

    # define the loss function
    model = NegativeSampling(
	    model = complEx, 
	    loss = SoftplusLoss(),
	    batch_size = train_dataloader.get_batch_size(), 
	    regul_rate = 1.0
    )

    # Train the model
    trainer = Trainer(model = model,
                    data_loader = train_dataloader,
                    train_times = 1000,
                    alpha = 0.5,
                    use_gpu = True
                    #opt_method="adagrad"
    )

    trainer.run()
    complEx.save_checkpoint('./checkpoint/complEx.ckpt')

    # Load the trained complEx model and export the embeddings
    complEx.load_checkpoint('./checkpoint/complEx.ckpt')

    # get the embeddings
    entity_embedding = complEx.ent_embeddings.weight.data.cpu().numpy()
    relation_embedding = complEx.rel_embeddings.weight.data.cpu().numpy()

    #save the embeddings
    np.savetxt('./output/entity_embeddings.txt', entity_embedding, delimiter='\t')
    np.savetxt('./output/relation_embeddings.txt', relation_embedding, delimiter='\t')


if __name__ == '__main__':
    
    main()
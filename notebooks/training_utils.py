import math
import random
import numpy
import torch
from typing import Callable
from torch.utils.data import Subset
from torch.utils.data.sampler import Sampler

from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

def chunk(indices, chunk_size=-1):
    if chunk_size<1:
        chunk_size = len(indices)
    return torch.split(torch.tensor(indices), chunk_size)


# Inspiration from https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#Samplers
class SequentialTrainingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False, base=10):
        max_segments = int(math.log(len(dataset), base))
        self.log_index_markers = [numpy.power(base, sgmt) for sgmt in range(1, max_segments+1)]
        if max_segments < math.log(len(dataset)):
            self.log_index_markers.append(len(dataset))
        self.indices_lists = [list(range(marker)) for marker in self.log_index_markers]
        self.shuffle = shuffle
        self.batch_size = batch_size
    
    def __iter__(self):
        if self.shuffle:
            for indices_list in self.indices_lists:
                random.shuffle(indices_list)
       ## print(self.indices_lists)
        segment_batches  = [chunk(segment_indices_list, self.batch_size) for segment_indices_list in self.indices_lists]

        #combined = [[batch.tolist() for batch in segment] for segment in segment_batches]
        combined = [batch.tolist() for segment in segment_batches for batch in segment]

        if self.shuffle:
            random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return  len(self.indices_lists)
    


def train_model_series_with_sequential_sampler(batch_size=32, n_epochs=0.1,
                                              compute_metrics: Callable = None):
    total_subset_idx = []
    sequential_supervision_val_scores = []
    sequential_supervision_test_scores = []
    for idx, idx_batch in enumerate(custom_sequential_sampler):
        if idx>4:
            break
    batch_dataset = Subset(encoded_dataset['train'], idx_batch)
    print(f"Number of training data points: {len(idx_batch)}")
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_NAME,
        logging_dir='./logs',            # directory for storing logs*
        logging_steps=2000,
        # report_to='wandb',
        save_total_limit = 5,
    )
    
    trainer = Trainer(
                model,
                args,
                train_dataset=batch_dataset,
                eval_dataset=encoded_dataset['valid'],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                #data_collator=data_collator
            )
    trainer.train()
    
    print(f"evaluating on validation set")
    trainer.eval_dataset=encoded_dataset['valid']
    val_scores = trainer.evaluate()        
    sequential_supervision_val_scores.append(val_scores)
    
    print(f"evaluating on test set")
    trainer.eval_dataset=encoded_dataset['test']
    test_scores = trainer.evaluate()
    sequential_supervision_test_scores.append(test_scores)
    return sequential_supervision_val_scores, sequential_supervision_test_scores
    #trainer.save()

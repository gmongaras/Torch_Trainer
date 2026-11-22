import os
cache_path = "./cache"
os.environ["HF_HOME"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path

import random
import datasets
import torch
from dataclasses import dataclass
import time
import threading
import queue
import concurrent.futures
from torch.utils.data import DataLoader
from typing import Optional
import math
from typing_extensions import Self
from src.utils import get_vocab_size

from src.types import DatasetName

TORCH_CPU = torch.device("cpu")






@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    cu_seqlens: torch.Tensor
    n_seqs: int
    
    def __post_init__(self):
        assert (
            len(self.input_ids.shape) 
            == len(self.labels.shape) 
            == len(self.cu_seqlens.shape) 
            == 1
        )
        assert self.input_ids.shape[0] == self.labels.shape[0]
        assert self.cu_seqlens[-1] == self.input_ids.shape[0]
        assert self.cu_seqlens[0] == 0



class BSSDataloader:
    def __init__(
            self, 
            dataset_name: DatasetName,
            local_bss: int,
            max_seqlen: int,
            min_train_seqlen: int,
            tokenizer,
            dataset_test_per: float,
            max_buffer_size: int,
            num_workers: int,
            rank: int,
            world_size: int,
            is_training: bool,
            dataset: Optional[datasets.Dataset] = None,
            block_load_size = 4096
        ):
        self.dataset_type = dataset_name
        self.local_bss = local_bss
        self.max_seqlen = max_seqlen
        self.min_train_seqlen = min_train_seqlen
        self.tokenizer = tokenizer
        self.dataset_test_per = dataset_test_per
        self.max_buffer_size = max_buffer_size
        self.num_workers = num_workers
        self.is_training = is_training
        self.dataset_ = dataset
        self.rank = rank
        self.world_size = world_size
        
        self.vocab_size = get_vocab_size(self.tokenizer)
        
        assert self.max_seqlen <= self.local_bss
        
        # Require a test if testing
        if not self.is_training:
            assert dataset is not None
        
        # Is this dataloader done processing data?
        # Should stay False when trianing
        self.is_done = False
        # Is the dataloader done processing data (is_done)
        # and out of data to server?
        self.is_complete = False
        
        # May change in the future, but if we are training
        # we are cool with partial final sequences. For
        # testing we are not.
        self.partial_sequences_ok = self.is_training
        
        # When we are not doing partial sequences,
        # we need to store the final batch we
        # retrieved that was too large in a thread
        # safe manner
        self.partial_storage = queue.Queue(
            maxsize=(
                1
                if self.partial_sequences_ok
                else self.num_workers
            )
        )
        
        # Buffer of data to process
        self.data_buffer = queue.Queue(maxsize=self.max_buffer_size)
        
        # Spin up thread to handle the dataloader
        thread = threading.Thread(
            target=self.init_head_thread,
            daemon=True,
        )
        thread.start()
        
        # Used to load in a block of contiguous data at a time
        self.block_load_size = block_load_size
        self.cur_block = None
        self.cur_block_idx = block_load_size
        
    
    def init_head_thread(self):
        # Load in the correct dataset
        self.load_hf_dataset()
        
        # Spin up some threads. They will always be working
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Data futures of running threads
            self.thread_futures = list()
            
            # Iter until we are done and there are no
            # more batches to retreive
            while (
                    not self.is_done 
                    or self.data_buffer.qsize() > 0
                    or len(self.thread_futures) > 0
                ):
                # If the total amount of data batches being created
                # is less than the max amount of data, submit more threads
                # to process more data
                if (
                        not self.is_done
                        and len(self.thread_futures) + self.data_buffer.qsize() < self.max_buffer_size
                    ):
                    self.thread_futures.append(
                        executor.submit(
                            self.prepare_batch,
                            self.is_training, # is_training
                        )
                    )
                
                # Check on all threads. If done, get the data
                # from the thread, store it, adn remove it
                # as a future.
                not_done = list()
                for f in self.thread_futures:
                    if f.done():
                        # Get results from thread
                        res, is_done = f.result()
                        # This should never be None when training
                        if self.is_training:
                            assert res is not None
                            assert not is_done
                        # If testing and done, we should
                        # not create any more batches.
                        # Mark the dataloader as done
                        else:
                            self.is_done = self.is_done or is_done
                        # Store batch
                        if res is not None:
                            self.data_buffer.put(res)
                    else:
                        not_done.append(f)
                self.thread_futures = not_done
                        
                time.sleep(1)
                
            # If we get to this point, there is no more data to process
            self.is_complete = True
                        

    def load_hf_dataset(self):
        # If dataset_ is not provided
        if self.dataset_ is None:
            # Load in train dataset
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            dataset_type, dataset_name = (
                self.dataset_type.value.split("%") 
                if "%" in self.dataset_type.value
                else (self.dataset_type.value, None)
            )
            self.dataset = datasets.load_dataset(
                dataset_type,
                name=dataset_name,
                cache_dir=cache_path,
                num_proc=16,
                split="train",
                download_config=datasets.DownloadConfig(
                    max_retries=20,
                    cache_dir=cache_path,
                )
            )
            
            # Load or create the test dataset
            try:
                # Does a test split exist?
                self.dataset_test = datasets.load_dataset(
                    dataset_type,
                    name=dataset_name,
                    cache_dir=cache_path,
                    num_proc=16,
                    split="test",
                    download_config=datasets.DownloadConfig(
                        max_retries=20,
                        cache_dir=cache_path,
                    )
                )
            except:
                # No shuffle as shuffle makes accessing the
                # dataset super slow.
                n = len(self.dataset)
                tst = int(n*self.dataset_test_per)
                trn = n - tst
                self.dataset_test = self.dataset.select(
                    range(trn, n)
                )
                self.dataset = self.dataset.select(
                    range(0, trn)
                )
                
            # Test dataset stuff. Note that this isn't actually
            # used but is here for returning the test dataloader
            # Convert to torch
            # self.dataset_test.set_format(type="torch", columns=["text"])
            # Dataset length
            self.num_sequences_test = len(self.dataset_test)
            
        # Dataset provided
        else:
            self.dataset = self.dataset_
        
        # Dataset lengths        
        self.num_sequences = len(self.dataset)
        
        # For testing, we need to shard the dataset over
        # all ranks
        if not self.is_training:
            # Get indices for this rank
            indices = list(range(0, self.num_sequences))
            rng = random.Random(123)
            rng.shuffle(indices)
            indices_chunk = math.ceil(self.num_sequences/self.world_size)
            indices_lo, indices_hi = indices_chunk*self.rank, indices_chunk*(self.rank+1)
            indices_local = indices[indices_lo:indices_hi]
            # Get subset of dataset
            self.dataset = self.dataset.select(indices_local)
            # Create dataloader and iterator
            self.test_data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x,
                num_workers=0,
            )
            self.test_data_iter = iter(self.test_data_loader)
            self.test_loader_lock = threading.Lock()
            self.num_sequences = len(self.dataset)
        
    @torch.no_grad()
    def get_seq(self, is_training):
        # First check the partial sequence queue for 
        # any sequences that were too long
        try:
            return self.partial_storage.get_nowait()
        except queue.Empty:
            pass
        
        # If trainig, sample a random sequence. To
        # make this faster, sample batches at a time
        if is_training:
            # Load new block
            if self.cur_block_idx == self.block_load_size:
                if self.num_sequences <= self.block_load_size:
                    blk_lo = 0
                    blk_hi = self.block_load_size
                else:
                    blk_lo = random.randint(0, self.num_sequences-self.block_load_size)
                    blk_hi = blk_lo+self.block_load_size
                self.cur_block = self.dataset[blk_lo:blk_hi]["text"]
                self.cur_block_idx = 0
            d = self.cur_block[self.cur_block_idx]
            self.cur_block_idx += 1
            return d
        
        # Testing samples from the dataloader. Return
        # None if no more data
        with self.test_loader_lock:
            try:
                return next(self.test_data_iter)[0]["text"]
            except StopIteration:
                return None
            
            
    # Helper to add an empty cu_seqlens
    # when the input_ids do not reach the end
    # of the sequence length. This can be due to:
    # - we do not accept partial batch, the last was partial
    # - in testing, we ran out of data
    # - last sequence was too short
    @torch.no_grad()
    def maybe_pad_cuseqlens(
            self,
            input_ids,
            cu_seqlens
        ):
        if cu_seqlens[-1] < input_ids.shape[0]:
            cu_seqlens.append(input_ids.shape[0])
    
    @torch.no_grad()
    def prepare_batch(
            self, 
            is_training,
        ):
        # Current batch and cu_seqlens
        # Default to the end of the vocab
        # for null ids.
        input_ids = torch.full(
            size=(self.local_bss,),
            fill_value=self.vocab_size-1,
            dtype=torch.long, 
            device=TORCH_CPU,
        )
        # Default to -100 to mask
        labels = torch.full(
            size=(self.local_bss,),
            fill_value=-100, 
            dtype=torch.long, 
            device=TORCH_CPU,
        )
        cu_seqlens = [0,]
        
        # is_done is True if we run out of data to process
        is_done = False
        
        # Iterate over sequences till we fill up the bss.
        n_seqs = 0
        while cu_seqlens[-1] < self.local_bss:
            # Get next random sequence
            random_seq = self.get_seq(is_training)
            
            # Testing is done when get_seq returns
            # the final batch
            if random_seq is None:
                assert not self.is_training
                is_done = True
                self.maybe_pad_cuseqlens(input_ids, cu_seqlens)
                break
            
            # Tokenize the random sequence. Chop
            # sequences that are way too long as they can
            # cause OOMs. Sequences greater than 5*max_seqlen
            # will be chopped to max_seqlen
            random_seq = random_seq[:10*self.max_seqlen]
            seq_enc = self.tokenizer.encode(
                random_seq, 
                add_special_tokens=True
            )
            
            # Remaining tokens this batch can take
            remaining_toks = self.local_bss-cu_seqlens[-1]
            
            # If partial sequences are not ok and the current
            # sequence is too long, store it for later processing
            usable_tokens = min(len(seq_enc) - 1, self.max_seqlen)
            if not self.partial_sequences_ok and usable_tokens > remaining_toks:
                self.partial_storage.put(random_seq)
                self.maybe_pad_cuseqlens(input_ids, cu_seqlens)
                break
            
            # If we do allow partial sequences, chop the
            # sequence to the remaining bss context.
            # Minus 1 for causal shift
            seq_len = min(min(
                self.local_bss-cu_seqlens[-1],
                self.max_seqlen
                )+1, len(seq_enc)
            )-1
            seq_enc = torch.tensor(
                seq_enc,
                dtype=input_ids.dtype
            )
            
            # Add to bss
            cu_seqlens.append(cu_seqlens[-1]+seq_len)
            input_ids[cu_seqlens[-2]:cu_seqlens[-1]] = seq_enc[:seq_len]
            
            # Add labels. Fill with -100 for no loss if the
            # sequence is too short
            if is_training and seq_len < self.min_train_seqlen:
                labels[cu_seqlens[-2]:cu_seqlens[-1]].fill_(-100)
            else:
                labels[cu_seqlens[-2]:cu_seqlens[-1]] = seq_enc[1:seq_len+1]
                
            n_seqs += 1
            
        # Edge case when we exit on the first iteration
        if len(cu_seqlens) == 1:
            return None, is_done
            
        return Batch(
            input_ids=input_ids,
            labels=labels,
            cu_seqlens=torch.tensor(
                cu_seqlens, 
                dtype=torch.long,
                device=TORCH_CPU
            ),
            n_seqs=n_seqs,
        ), is_done

    # Main thread calls this
    @torch.no_grad()
    def sample_batch(self) -> Batch:
        # None is returned if the dataloader is
        # fully out of data
        if self.is_complete:
            return None
        
        # Otherwise we got data to give
        batch = None
        while batch is None:
            if self.is_complete:
                return None
            try:
                batch = self.data_buffer.get_nowait()
            except queue.Empty:
                pass
            time.sleep(0.5)
        return batch
        
    # Returns a new test dataloader to start sampling from
    @torch.no_grad()
    def get_test_dataloader(self) -> Self:
        return BSSDataloader(
            dataset_name=self.dataset_type,
            local_bss=self.local_bss,
            max_seqlen=self.max_seqlen,
            min_train_seqlen=self.min_train_seqlen,
            tokenizer=self.tokenizer,
            dataset_test_per=self.dataset_test_per,
            max_buffer_size=self.max_buffer_size,
            num_workers=self.num_workers,
            rank=self.rank,
            world_size=self.world_size,
            is_training=False,
            dataset=self.dataset_test,
        )
    
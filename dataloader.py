from time import time
from datetime import timedelta
import json

from fastNLP.io.loader import JsonLoader
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.pipe.pipe import Pipe
from fastNLP.core.const import Const
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer

class MatchSumLoader(JsonLoader):
    def __init__(self, candidate_num, encoder, max_len=180):
        super(MatchSumLoader, self).__init__(fields={})  # Fix: Initialize with fields={}
        self.candidate_num = candidate_num
        self.max_len = max_len
        self.encoder = encoder
        
        self.tokenizer = AutoTokenizer.from_pretrained(encoder)
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def _load(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                src = example["src"]
                candidates = example["candidates"][:self.candidate_num]
                summary = example["summary"]
                
                src_ids = self.tokenizer.encode(
                    src, 
                    max_length=self.max_len,
                    truncation=True,
                    padding='max_length',
                    add_special_tokens=True
                )
                
                candidate_ids = []
                for cand in candidates:
                    cand_ids = self.tokenizer.encode(
                        cand, 
                        max_length=self.max_len,
                        truncation=True,
                        padding='max_length',
                        add_special_tokens=True
                    )
                    candidate_ids.append(cand_ids)
                
                instance = Instance(
                    text_id=src_ids,
                    candidate_id=candidate_ids,
                    summary_id=candidate_ids[0],
                    text=src,
                    summary=summary
                )
                data.append(instance)
        return DataSet(data)

    def load(self, paths):
        print('Start loading datasets !!!')
        start = time()

        datasets = {}
        for name in paths:
            dataset = self._load(paths[name])
            dataset.set_input("text_id", "candidate_id", "summary_id")
            dataset.set_target("summary_id")
            dataset.set_pad_val("text_id", self.pad_token_id)
            dataset.set_pad_val("candidate_id", self.pad_token_id)
            dataset.set_pad_val("summary_id", self.pad_token_id)
            datasets[name] = dataset

        print('Finished in {}'.format(timedelta(seconds=time()-start)))
        return DataBundle(datasets=datasets)

class MatchSumPipe(Pipe):
    def __init__(self, candidate_num, encoder):
        super(MatchSumPipe, self).__init__()
        self.candidate_num = candidate_num
        self.encoder = encoder

    def process(self, data_bundle):
        return data_bundle
        
    def process_from_file(self, paths):
        data_bundle = MatchSumLoader(self.candidate_num, self.encoder).load(paths)
        return self.process(data_bundle)
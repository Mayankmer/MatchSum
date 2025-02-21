import os
import re
import argparse
from os.path import join, exists
import subprocess as sp
import json
import tempfile
import multiprocessing as mp
from time import time
from datetime import timedelta
import queue
import logging
from itertools import combinations

from cytoolz import curry
from pyrouge.utils import log
from pyrouge import Rouge155
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Legal-specific configuration
MAX_LEN = 1024  # Increased for legal document length
_LEGAL_CITATION_PATTERN = r'\b(AIR \d{4} (SC|Ker|Bom|Cal|Mad|All) \d+\b'
_ROUGE_PATH = '/path/to/RELEASE-1.5.5'
temp_path = './legal_temp'  # Changed temp path for legal processing

original_data, sent_ids = [], []

def preprocess_legal_text(text):
    """Clean and normalize legal text elements"""
    text = re.sub(_LEGAL_CITATION_PATTERN, '[CITATION]', text)
    text = re.sub(r'Sec(?:tion)?\.?\s+\d+[A-Za-z]*', '[SECTION]', text)
    text = re.sub(r'\b(?:Petitioner|Respondent)s?\b', '[PARTY]', text)
    return text

def load_legal_data(data_path):
    """Load IN-ABS dataset with legal preprocessing"""
    dataset = load_dataset("percins/IN-ABS", split='train')
    processed = []
    for case in dataset:
        processed.append({
            'case': [preprocess_legal_text(sent) for sent in case['text']],
            'summary': [preprocess_legal_text(sent) for sent in case['summary']]
        })
    return processed

@curry
def get_candidates(tokenizer, cls, sep_id, idx):
    idx_path = join(temp_path, str(idx))
    
    # Create temporary directory structure
    sp.call(f'mkdir -p {idx_path}', shell=True)
    sp.call(f'mkdir -p {join(idx_path, "decode")}', shell=True)
    sp.call(f'mkdir -p {join(idx_path, "reference")}', shell=True)
    
    # Load legal case data
    data = {
        'case': original_data[idx]['case'],
        'summary': original_data[idx]['summary']
    }
    
    # Write reference summary
    ref_dir = join(idx_path, 'reference')
    with open(join(ref_dir, '0.ref'), 'w') as f:
        for sentence in data['summary']:
            print(sentence, file=f)

    # Legal-specific candidate generation
    sent_id = sent_ids[idx]['sent_id'][:8]  # Use top 8 sentences
    indices = list(combinations(sent_id, 3))  # Minimum 3 sentences
    indices += list(combinations(sent_id, 4))  # Add 4-sentence combinations
    
    if len(sent_id) < 3:  # Handle short documents
        indices = [sent_id] if len(sent_id) > 0 else []

    # Score candidates with legal-aware ROUGE
    score = []
    for i in indices:
        i = list(i)
        i.sort()
        dec = [data['case'][j] for j in i]
        score.append((i, get_rouge(idx_path, dec)))
    score.sort(key=lambda x: x[1], reverse=True)

    # Store legal document metadata
    data['court'] = original_data[idx].get('court', '')
    data['year'] = original_data[idx].get('year', '')
    data['ext_idx'] = sent_id
    data['indices'] = [list(map(int, i)) for i, _ in score]
    data['score'] = [r for _, r in score]

    # Legal-specific tokenization
    candidate_summary = []
    for i in data['indices']:
        cur_summary = [cls]
        for j in i:
            cur_summary += data['case'][j].split()
        cur_summary = cur_summary[:MAX_LEN]
        candidate_summary.append(' '.join(cur_summary))
    
    data['candidate_id'] = []
    for summary in candidate_summary:
        token_ids = tokenizer.encode(
            summary, 
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN-1
        ) + sep_id
        data['candidate_id'].append(token_ids)

    # Tokenize full case text
    case_text = [cls]
    for sent in data['case']:
        case_text += sent.split()
    case_text = ' '.join(case_text[:MAX_LEN])
    data['text_id'] = tokenizer.encode(
        case_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN-1
    ) + sep_id

    # Tokenize reference summary
    ref_summary = [cls]
    for sent in data['summary']:
        ref_summary += sent.split()
    ref_summary = ' '.join(ref_summary[:MAX_LEN])
    data['summary_id'] = tokenizer.encode(
        ref_summary,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN-1
    ) + sep_id

    # Save processed legal case
    processed_path = join(temp_path, 'processed')
    with open(join(processed_path, f'{idx}.json'), 'w') as f:
        json.dump(data, f, indent=4)
    
    sp.call(f'rm -rf {idx_path}', shell=True)

def get_candidates_mp(args):
    # Initialize legal tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    sep = tokenizer.sep_token
    sep_id = tokenizer.convert_tokens_to_ids(sep)
    cls = tokenizer.cls_token

    # Load legal data
    global original_data, sent_ids
    original_data = load_dataset("percins/IN-ABS", split='train')
    sent_ids = load_dataset(args.index_path, split='train')
    
    print(f'Processing {len(original_data)} legal cases')
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(join(temp_path, 'processed'), exist_ok=True)

    # Parallel processing for legal documents
    start = time()
    print('Starting legal candidate generation...')
    
    with mp.Pool() as pool:
        pool.imap_unordered(
            get_candidates(tokenizer, cls, sep_id),
            range(len(original_data)),
            chunksize=32  # Reduced for larger documents
        )
    
    print(f'Completed in {timedelta(seconds=time()-start)}')
    
    # Aggregate results for legal dataset
    print('Compiling final dataset...')
    with open(args.write_path, 'w') as outfile:
        for i in range(len(original_data)):
            with open(join(temp_path, 'processed', f'{i}.json')) as f:
                json.dump(json.load(f), outfile)
                outfile.write('\n')
    
    sp.call(f'rm -rf {temp_path}', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate candidate summaries for legal cases'
    )
    parser.add_argument('--data_path', type=str, required=True,
        help='Path to IN-ABS dataset directory')
    parser.add_argument('--index_path', type=str, required=True,
        help='Path to sentence indices file')
    parser.add_argument('--write_path', type=str, required=True,
        help='Output path for processed legal candidates')
    
    args = parser.parse_args()
    # assert exists(args.data_path), "Dataset path not found"
    # assert exists(args.index_path), "Index file not found"
    
    get_candidates_mp(args)
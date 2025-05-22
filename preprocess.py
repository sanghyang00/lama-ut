import pandas as pd
import numpy as np
import uroman as ur
from tqdm import tqdm
from datasets import load_dataset
import re, os, string, argparse, csv, unicodedata
import pykakasi
from mappings import lang_to_iso, cv_to_iso, cv_to_lang

def contains_parentheses_and_digits(text):
    pattern = r'[()\[\]{}0-9]'
    if re.search(pattern, text):
        return True
    else:
        return False

def normalize_transcription(text):
    special_chars = string.punctuation.replace("'","")
    pattern = f"[{re.escape(special_chars)}]"
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(pattern, '', text)
    return text

def romanize_japanese(text, kks):
    special_chars = string.punctuation.replace("'","")
    pattern = f"[{re.escape(special_chars)}]"
    romanized = ''.join([i['hepburn'] for i in kks.convert(text)])
    romanized = re.sub(pattern, '', romanized)
    return romanized

def check_invalid(text):
    pattern = r'[^a-z\' ]'
    is_invalid = bool(re.search(pattern, text))
    return is_invalid

def sanity_check(df, column_name):
    pattern = re.compile(r"^[a-z' ]*$")
    
    for value in df[column_name]:
        if not pattern.match(value):
            return False
    
    return True

def preprocess_cv(tsv_dir, split='train'):
    
    save_path = 'preprocessed_cv'
    os.makedirs(save_path, exist_ok=True)

    result = pd.DataFrame()
    
    # Configuration of 25 unseen languages. 
    # 'rm-sursilv' and 'rm-vallader' are both considered as Romansh.
    unseen_languages = ['ab', 'sq', 'bas', 'ba', 'eu',
                        'br', 'cv', 'mhr', 'myv', 'eo',
                        'gn', 'cnh', 'ia', 'rw', 'ltg',
                        'nn-NO', 'rm-sursilv', 'rm-vallader',
                        'tt', 'tok', 'tk', 'ug', 'hsb',
                        'fy-NL', 'mrj', 'sah']
    
    # Merge separate test.tsv files into single DataFrame.
    for region in tqdm(os.listdir(tsv_dir)):
        if region in unseen_languages:
            subfolder_path = os.path.join(tsv_dir, region)
            audio_dir = os.path.join(subfolder_path, 'clips')
            data = pd.read_csv(os.path.join(subfolder_path, f'{split}.tsv'), sep='\t', 
                                    low_memory=False, on_bad_lines='skip', quoting=csv.QUOTE_NONE)
            data['path'] = data['path'].apply(lambda x: os.path.join(audio_dir, x))
            data['language'] = data['locale'].apply(lambda x: cv_to_lang[region])
            
            # For the training split, the number of samples is limited: only used for shot generation.
            # Remove the following two lines if needed.
            if len(data) > 500 and split == 'train':
                data = data.sample(500)
            
            result = pd.concat([result, data], axis=0)
            
    # Remove samples that cause ambiguity in pronunciation
    result = result[~result['sentence'].apply(contains_parentheses_and_digits)]
    result = result[['path', 'sentence', 'locale', 'language']]
    result['transcription'] = result['sentence'].apply(normalize_transcription)
    
    # Apply Romanization to merged data
    uroman = ur.Uroman()
    transcriptions = result['transcription'].tolist()
    codes = result['locale'].tolist()
    romans = []
    for (transcription, code) in tqdm(zip(transcriptions, codes), total=len(transcriptions)):
        transcription = str(transcription)
        romanized = uroman.romanize_string(transcription, lcode=cv_to_iso[code])
        romanized = unicodedata.normalize('NFKC', romanized)
        romans.append(romanized)
    
    result['roman'] = romans
    
    # Prune transcriptions that have invalid values (e.g., characters except for <lowercased alphabet, spacing and apostrophe>)
    result = result[~result['roman'].apply(check_invalid)].reset_index(drop=True)
    
    # Sort only the necessary columns and rename 
    result = result.rename(columns={'sentence': 'raw transcription', 'path': 'audio path'})
    result = result[['audio path', 'raw transcription', 'transcription', 'language', 'roman']]
    
    assert sanity_check(result, 'roman')
    
    # Save the preprocessed dataset as a tsv file
    result.to_csv(os.path.join(save_path, f'{split}.tsv'), sep='\t', index=False)

def preprocess_fleurs(ds, split='train'):
    save_path = 'preprocessed_fleurs'
    os.makedirs(save_path, exist_ok=True)
    
    # Organize the necessary information into DataFrame (huggingface dataset -> pandas DataFrame)
    audio_paths = []
    raw_transcriptions = []
    transcriptions = []
    languages = []
    lang_group_ids = []
    durations = []
    for i in tqdm(range(len(ds))):
        audio_paths.append(os.path.join(os.path.dirname(ds[i]['path']), ds[i]['audio']['path']))
        raw_transcriptions.append(ds[i]['raw_transcription'])
        transcriptions.append(ds[i]['transcription'])
        languages.append(ds[i]['language'])
        lang_group_ids.append(ds[i]['lang_group_id'])
        durations.append(round(ds[i]['num_samples']/16000, 4))
        
    result = pd.DataFrame()
    result['audio path'] = audio_paths
    result['raw transcription'] = raw_transcriptions
    result['transcription'] = transcriptions
    result['language'] = languages
    result['lang_group_id'] = lang_group_ids
    result['duration'] = durations
    
    # Remove samples that cause ambiguity in pronunciation
    result = result[~result['raw transcription'].apply(contains_parentheses_and_digits)]
    # Prune excessively long samples
    result = result[result['duration'] < 25]
    
    assert result.isna().sum().sum() == 0
    assert (result['transcription'] == '').sum() == 0
    
    # Sort only the necessary columns 
    result['transcription'] = result['transcription'].apply(normalize_transcription)
    result = result[['audio path', 'raw transcription', 'transcription', 'language']]
    
    # Apply Romanization to merged data
    uroman = ur.Uroman()
    kks = pykakasi.kakasi()
    transcriptions = result['transcription'].tolist()
    languages = result['language'].tolist()
    romans = []
    for (transcription, lang) in tqdm(zip(transcriptions, languages), total=len(transcriptions)):
        if (lang=='Japananese'):
            romanized = romanize_japanese(transcription, kks)
        else:
            transcription = str(transcription)
            romanized = uroman.romanize_string(transcription, lcode=lang_to_iso[lang])
        romanized = unicodedata.normalize('NFKC', romanized)
        romans.append(romanized)
    
    result['roman'] = romans
    
    # Prune transcriptions that have invalid values (e.g., characters except for <lowercased alphabet, spacing and apostrophe>)
    result = result[~result['roman'].apply(check_invalid)].reset_index(drop=True)
    
    # Rearrange columns
    result = result[['audio path', 'raw transcription', 'transcription', 'language', 'roman']]
    
    assert sanity_check(result, 'roman')
    
    # Save the preprocessed dataset as a tsv file
    result.to_csv(os.path.join(save_path, f'{split}.tsv'), sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATASET', type=str, default='FLEURS')
    parser.add_argument('--CV_PATH', type=str, default=None) # Path to CommonVoice dataset folder, which contains subfolders of each region
    
    args = parser.parse_args()
    
    # Preprocess FLEURS data for seen languages.  
    if args.DATASET == 'FLEURS':
        fleurs = load_dataset('google/fleurs', 'all', cache_dir='/ssd/sangmin/datasets/fleurs', trust_remote_code=True) # Add your save path
        
        fleurs_train = fleurs['train']
        fleurs_dev = fleurs['validation']
        fleurs_test = fleurs['test']
        
        print('Preprocessing Train Split')
        preprocess_fleurs(fleurs_train, split='train')   
        print('Preprocessing Dev Split')    
        preprocess_fleurs(fleurs_dev, split='dev')   
        print('Preprocessing Test Split')     
        preprocess_fleurs(fleurs_test, split='test')
        
    # Preprocess CommonVoice data for unseen languages.
    elif args.DATASET == 'CV':
        tsv_dir = args.CV_PATH
        print('Preprocessing Train Split')
        preprocess_cv(tsv_dir, split='train') # For shot generation
        print('Preprocessing Test Split')
        preprocess_cv(tsv_dir, split='test')
    
    else:
        raise ValueError('DATASET value must be in [\'FLEURS\', \'CV\']')

if __name__=='__main__':
    main()
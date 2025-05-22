import torch
import torch.nn as nn
import torchaudio
from torchaudio.models.decoder import ctc_decoder
import pandas as pd
import argparse, json, os, re
from tqdm import tqdm
from openai import OpenAI

class translator():
    def __init__(self, vocab_path):
        self.roman_to_label = load_dictionary(vocab_path)
        self.label_to_roman = flip_dictionary(self.roman_to_label)
        
    def to_label(self, roman):
        
        labels = []
        
        for character in roman:
            labels.append(self.roman_to_label[str(character)])
            
        return labels
    
    def to_transcription(self, labels):
        transcription = []
        for label in labels:
            transcription.append(self.label_to_roman[int(label)])
        
        transcription = ''.join(transcription)    
        
        return transcription

def load_dictionary(dict_path):
    with open(dict_path, 'r') as file:
        dictionary = json.load(file)
        
    return dictionary

def flip_dictionary(dictionary):
    
    flipped_dictionary = {v: k for k, v in dictionary.items()}
    
    return flipped_dictionary

def normalize_audio(x):
    max_value = torch.max(torch.abs(x))

    normalized_x = x / max_value
    
    return normalized_x

def generate_shots(data, language, num_shots=5):
    data = data[data['language']==language]
    sampled_data = data.sample(n=num_shots)
    shots = []
    
    for i in range(num_shots):
        shot = f"{sampled_data.iloc[i]['roman']} => {sampled_data.iloc[i]['raw transcription']}\n"
        shots.append(shot)

    generated_shots = "".join(shots)
    
    return generated_shots

def sort_output(output):
    pattern = r'```([^`]*)```'
    sorted_output = re.findall(pattern, output)
    
    if len(sorted_output) > 0:
        return sorted_output[0]
    else:
        return ''

def main():
    parser = argparse.ArgumentParser()
    # For inference of unseen data, substitute this with preprocessed_cv/train.tsv
    parser.add_argument('--TRAIN_DATA_PATH', type=str, default='preprocessed_fleurs/train.tsv') 
    # For inference of unseen data, substitute this with preprocessed_cv/test.tsv
    parser.add_argument('--TEST_DATA_PATH', type=str, default='preprocessed_fleurs/test.tsv')
    parser.add_argument('--VOCAB_PATH', type=str, default='vocab.json')
    parser.add_argument('--UTG_SIZE', type=str, default='1b')
    parser.add_argument('--CKPT_PATH', type=str, default=None)
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--OPENAI_API_KEY', type=str, default=None)
    parser.add_argument('--PROMPT_TYPE', type=str, default='FEWSHOT')
    
    args = parser.parse_args()
    
    # Load train/test data. Train data is used in shot generation.
    train_data = pd.read_csv(args.TRAIN_DATA_PATH, sep='\t')
    test_data = pd.read_csv(args.TEST_DATA_PATH, sep='\t')
    
    # Decoder setup
    t = translator(args.VOCAB_PATH)
    vocab = load_dictionary(args.VOCAB_PATH)
    labels = list(vocab.keys())
    decoder = ctc_decoder(
        lexicon=None,
        tokens=labels,
        lm=None,
        beam_size=100,
        blank_token='|',
        sil_token=' '
    )
    
    # Load Universal Transcription Generator
    if args.UTG_SIZE == '300m':
        model =  torchaudio.models.wav2vec2_xlsr_300m(aux_num_out=len(vocab))
    
    elif args.UTG_SIZE == '1b':
        model = torchaudio.models.wav2vec2_xlsr_1b(aux_num_out=len(vocab))
    
    else:
        raise ValueError('<UTG_SIZE> value must be in [\'300m\', \'1b\'].')
    
    # Load Checkpoint
    try:
        model.load_state_dict(torch.load(args.CKPT_PATH))
    
    except:
        raise ValueError('<CKPT_PATH> must be a valid path.')
    
    model.eval()
    model.to(args.DEVICE)
    
    for i in tqdm(range(len(test_data))):
        
        audio_path = test_data.iloc[i]['audio path']
        lang = test_data.iloc[i]['language']
        gt = test_data.iloc[i]['raw transcription']
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
        audio = normalize_audio(audio)
        
        # Universal Transcription Generator Inference
        with torch.no_grad():
            logits, mask = model(audio.to(args.DEVICE))
            logits = logits.detach().cpu()
            
        logits = nn.functional.log_softmax(logits, dim=-1)
        hypothesis = decoder(logits)
        roman = t.to_transcription(hypothesis[0][0][0]).strip()
            
        # Universal Converter Inference (converter model can be changed)
        os.environ['OPENAI_API_KEY'] = args.OPENAI_API_KEY
        client = OpenAI()
        
        if args.PROMPT_TYPE == 'ZEROSHOT':
            completion = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": f"You are a translation expert who translates Romanized transcriptions into {lang}. Final transcription must be given between three backticks. Example: ```example```"},
                    {"role": "user", "content": f"Transcribe following Romanized sentence into a {lang} sentence: {roman}."}
                ],
                temperature=0.0,
            )
            
            raw_pred = completion.choices[0].message.content
            pred = sort_output(raw_pred)
            
        elif args.PROMPT_TYPE == 'FEWSHOT':
            
            max_samples = len(train_data[train_data['language']==lang])
            
            if max_samples == 0:
                shots = ''
                
            elif max_samples < 5:
                shots = generate_shots(train_data, lang, num_shots=max_samples)
                
            else:
                shots = generate_shots(train_data, lang, num_shots=5)
                
            completion = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": f"You are a translation expert who translates Romanized transcriptions into {lang}. Final transcription must be given between three backticks. Example: ```example```"},
                    {"role": "user", "content": f"""Here are some examples of transcribing a Romanized sentence into a {lang} sentence: 
                     {shots}
                     Considering the examples above, transcribe the following Romanized sentence into a {lang} sentence: {roman}."""}
                ],
                temperature=0.0,
            )
            
            raw_pred = completion.choices[0].message.content
            pred = sort_output(raw_pred)
            
        elif args.PROMPT_TYPE == 'COT':
            completion = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": f"You are a translation expert who translates Romanized transcriptions into {lang}. Final transcription must be given between three backticks. Example: ```example```"},
                    {"role": "user", "content": f"Transcribe the following Romanized sentence into a {lang} sentence. Think step by step: {roman}."}
                ],
                temperature=0.0,
            )
            
            raw_pred = completion.choices[0].message.content
            pred = sort_output(raw_pred)
            
        elif args.PROMPT_TYPE == 'FEWSHOT+COT':     
                   
            max_samples = len(train_data[train_data['language']==lang])
            
            if max_samples == 0:
                shots = ''
                
            elif max_samples < 5:
                shots = generate_shots(train_data, lang, num_shots=max_samples)
                
            else:
                shots = generate_shots(train_data, lang, num_shots=5)
                
            completion = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": f"You are a translation expert who translates Romanized transcriptions into {lang}. Final transcription must be given between three backticks. Example: ```example```"},
                    {"role": "user", "content": f"""Here are some examples of transcribing a Romanized sentence into a {lang} sentence: 
                     {shots}
                     Considering the examples above, transcribe the following Romanized sentence into a {lang} sentence. Think step by step: {roman}."""}
                ],
                temperature=0.0,
            )
            
            raw_pred = completion.choices[0].message.content
            pred = sort_output(raw_pred)
        
        elif args.PROMPT_TYPE == 'CHAINING':
            completion1 = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": f"You are a translation expert who translates Romanized transcriptions into {lang}. Final transcription must be given between three backticks. Example: ```example```"},
                    {"role": "user", "content": f"""Transcribe the following Romanized sentence into a {lang} sentence, based on its pronunciation: {roman}."""}
                ],
                temperature=0.0,
            )
            
            raw_pred1 = completion1.choices[0].message.content
            pred1 = sort_output(raw_pred1)
            
            completion2 = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": f"You are a translation expert who translates Romanized transcriptions into {lang}. Final transcription must be given between three backticks. Example: ```example```"},
                    {"role": "user", "content": f"""Correct the typographical and spacing errors in the following {lang} sentence: {pred1}."""}
                ],
                temperature=0.0,
            )
            
            raw_pred2 = completion2.choices[0].message.content
            pred = sort_output(raw_pred2)
        
        else: 
            raise ValueError('Valid prompt types: ZEROSHOT, FEWSHOT, COT, FEWSHOT+COT, CHAINING')
        
        # Print ground truth and final prediction
        print(f'GT: {gt}')
        print(f'PRED: {pred}')
    

if __name__=='__main__':
    main()
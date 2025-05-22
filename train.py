import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from scheduler import TriStageLRScheduler
import librosa
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm
import argparse
import os
import json
import re

MINIBATCH = 0
NUM_ACCUMULATION_STEPS = 0
CURRENT_STEP = 0
SPECIFIC_STEP = 0
NUM_TOTAL_STEPS = 3000
torch.autograd.set_detect_anomaly(True)

class Fleurs(Dataset):
    def __init__(self, folder_path, mode='train'):
        
        assert (mode in ['train', 'dev', 'test'])
        self.tsv_path = os.path.join(folder_path, f'{mode}.tsv')
        self.data = pd.read_csv(self.tsv_path, sep='\t')
        self.data = self.data.dropna().reset_index(drop=True)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['audio path']
        audio, sr = torchaudio.load(audio_path)
        assert (sr == 16000)
        audio = normalize_audio(audio)
        
        if audio.isnan().any():
            
            print(f'NaN detected in audio file: {audio_path}')
            print(f'Replaced audio and transcription into blank values')
            
            return torch.zeros(1, 1600, dtype=torch.float32), ''
        
        roman = self.data.iloc[idx]['roman']
        
        return audio, roman
    
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

def collate_fn(batch, vocab_path=None):
    
    vocab_path = 'vocab.json'
    inputs = []
    labels = []
    input_lengths = []
    label_lengths = []
    padding_value = 0 # Blank token index
    
    t = translator(vocab_path)
    
    for (audio, roman) in batch:
        audio = rearrange(audio, 'b l -> (b l)')
        inputs.append(audio)
        label = t.to_label(roman)
        label = torch.tensor(label)
        labels.append(label)
        input_lengths.append(audio.size(0))
        label_lengths.append(len(label))
        
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=padding_value)
    
    return inputs, labels, input_lengths, label_lengths

def train(model, train_loader, criterion, optimizer, device, scheduler=None):
    
    global NUM_ACCUMULATION_STEPS
    global CURRENT_STEP
    global SPECIFIC_STEP
    global NUM_TOTAL_STEPS
    
    model.train()
    
    running_loss = 0
    step_loss = 0
    save_dir = f'ckpts'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (inputs, labels, input_lengths, label_lengths) in enumerate(train_loader):
        
        inputs, labels, input_lengths, label_lengths = inputs.to(device), labels.to(device), torch.tensor(input_lengths).to(device), torch.tensor(label_lengths).to(device)

        outputs, input_lengths = model(inputs, input_lengths)
        outputs = F.log_softmax(outputs, dim=2) # Shape: B L C
        outputs = rearrange(outputs, 'b l c -> l b c') # Shape: L B C -> Input specification of CTCLoss
        
        loss = criterion(outputs, labels, input_lengths, label_lengths)

        loss = loss / NUM_ACCUMULATION_STEPS # Gradient accumulation
        loss.backward()

        running_loss += loss.item()
        step_loss += loss.item()
        SPECIFIC_STEP += 1
        
        del loss, inputs, labels, input_lengths, label_lengths

        if ((i+1) % NUM_ACCUMULATION_STEPS == 0) or (i+1 == len(train_loader)):
            
            if scheduler is not None:
               scheduler.step()
               
            CURRENT_STEP += 1
            
            print(f'Step {CURRENT_STEP} | Loss: {step_loss:.4f}')
            
            optimizer.step()
            optimizer.zero_grad()
            step_loss = 0
            
        if SPECIFIC_STEP % (NUM_ACCUMULATION_STEPS * 500) == 0:
            save_path = os.path.join(save_dir, f'iter{CURRENT_STEP}.pt')
            torch.save(model.state_dict(), save_path)        
        
    epoch_loss = running_loss * NUM_ACCUMULATION_STEPS / len(train_loader)
    
    return epoch_loss

def evaluate(model, val_loader, criterion, device):
    
    model.eval()
    running_loss = 0

    for i, (inputs, labels, input_lengths, label_lengths) in enumerate(val_loader):
        
        print(f'Evaluation {i}/{len(val_loader)}')
        
        inputs, labels, input_lengths, label_lengths = inputs.to(device), labels.to(device), torch.tensor(input_lengths).to(device), torch.tensor(label_lengths).to(device)

        with torch.no_grad():
            outputs, input_lengths = model(inputs, input_lengths)
            
        outputs = F.log_softmax(outputs, dim=2) # Shape: B L C

        outputs = rearrange(outputs, 'b l c -> l b c') # Shape: L B C -> Input specification of CTCLoss

        loss = criterion(outputs, labels, input_lengths, label_lengths)

        running_loss += loss.item()
        
        del loss, inputs, labels, input_lengths, label_lengths
        
    epoch_loss = running_loss / len(val_loader)
            
    return epoch_loss

def run(model, num_epoch=100, optimizer=None, scheduler=None, data_dir=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 16
    criterion = nn.CTCLoss(zero_infinity=True)
    
    train_dataset = Fleurs(data_dir, mode='train')
    val_dataset = Fleurs(data_dir, mode='dev')
    
    train_loader = DataLoader(train_dataset,
                          batch_size=MINIBATCH,
                          shuffle=True,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=MINIBATCH,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)
    
    model.to(device)
    
    for i in range(num_epoch):                    
        train_loss = train(model, train_loader, criterion, optimizer, device, scheduler)
        # val_loss = evaluate(model, val_loader, criterion, device) # Uncomment it if needed

def main():

    global NUM_ACCUMULATION_STEPS
    global MINIBATCH
    global NUM_TOTAL_STEPS
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='1b')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--minibatch', type=int, default=1)
    parser.add_argument('--total_step', type=int, default=3000)
    parser.add_argument('--init_lr', type=float, default=5e-6)
    parser.add_argument('--peak_lr', type=float, default=5e-4)
    parser.add_argument('--final_lr', type=float, default=5e-6)
    parser.add_argument('--data_dir', type=str, default='preprocessed_fleurs')
    
    args = parser.parse_args()
    
    assert args.batch % args.minibatch == 0
    
    MINIBATCH = args.minibatch
    NUM_ACCUMULATION_STEPS = args.batch / args.minibatch
    NUM_TOTAL_STEPS = args.total_step

    vocab_path = 'vocab.json' # Vocabulary file path (.json)
    vocab = load_dictionary(vocab_path)
    data_dir = args.data_dir # Preprocessed dataset directory (folder path that contains tsv data)
    num_classes = len(vocab)
    
    if args.model_size == '300m':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        ssl_model = bundle.get_model().model
        fe = ssl_model.feature_extractor.state_dict()
        enc = ssl_model.encoder.state_dict()

        model = torchaudio.models.wav2vec2_xlsr_300m(
            encoder_projection_dropout=0.1,
            encoder_attention_dropout=0.1,
            encoder_ff_interm_dropout=0.1,
            encoder_dropout=0.1,
            encoder_layer_drop=0.1,
            aux_num_out=num_classes
            )
        
    elif args.model_size == '1b':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
        ssl_model = bundle.get_model().model
        fe = ssl_model.feature_extractor.state_dict()
        enc = ssl_model.encoder.state_dict()

        model = torchaudio.models.wav2vec2_xlsr_1b(
            encoder_projection_dropout=0.1,
            encoder_attention_dropout=0.1,
            encoder_ff_interm_dropout=0.1,
            encoder_dropout=0.1,
            encoder_layer_drop=0.1,
            aux_num_out=num_classes
            )
        
    else:
        raise ValueError('Model size value must be in [\'300m\', \'1b\']')
        
    model.feature_extractor.load_state_dict(fe)
    model.encoder.load_state_dict(enc)

    for name, param in model.feature_extractor.named_parameters():
        param.requires_grad=False
    
    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)
    num_warmup_steps = int(NUM_TOTAL_STEPS * 0.1)
    num_hold_steps = int(NUM_TOTAL_STEPS * 0.6)
    num_decay_steps = NUM_TOTAL_STEPS - num_warmup_steps - num_hold_steps        
    scheduler = TriStageLRScheduler(optimizer, init_lr=args.init_lr, peak_lr=args.peak_lr, final_lr=args.final_lr, 
                            warmup_steps=num_warmup_steps, hold_steps=num_hold_steps, decay_steps=num_decay_steps, total_steps=NUM_TOTAL_STEPS)

    run(model, num_epoch=100, optimizer=optimizer, scheduler=scheduler, data_dir=data_dir)
    
if __name__=='__main__':
    main()
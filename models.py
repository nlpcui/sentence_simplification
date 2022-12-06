import transformers, torch, math, logging, argparse, random, os
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, BartTokenizer, AutoConfig, AutoTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    

class SimplificationDataset(Dataset):
    def __init__(self, original, simplified, tokenizer, prepare_decoder_input_ids_from_labels, max_length, padding='max_length', truncation=True, return_tensors='pt'):
        self.input_ids = []
        self.attn_masks = []
        self.label_ids = []
        self.decoder_input_ids = []

        for i in range(len(simplified)):
            original_encoded = tokenizer(original[i], max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)
            simplified_encoded = tokenizer(simplified[i], max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)
        
            self.input_ids.append(original_encoded['input_ids'][0])
            self.attn_masks.append(original_encoded['attention_mask'][0])
            label_ids = simplified_encoded['input_ids']
            label_ids[label_ids==tokenizer.pad_token_id] = -100
            self.label_ids.append(label_ids[0])
            self.decoder_input_ids.append(prepare_decoder_input_ids_from_labels(label_ids)[0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.label_ids[idx], self.decoder_input_ids[idx]


def train_bart(args, original, simplified, model_save_path, device):
    bart_tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)
    
    train_dataset = SimplificationDataset(original, simplified, bart_tokenizer, model.prepare_decoder_input_ids_from_labels, args.max_length)
    logging.info('train size {}'.format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_total_steps = len(train_dataloader) * args.epoch
    warmup_steps = int(train_total_steps * args.warmup_rate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_total_steps
    )

    for epoch_idx in range(args.epoch):
        # train
        train_loss = 0
        model.train()
        for batch_idx, (input_ids, attn_mask, label_ids, decoder_input_ids) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device), labels=label_ids.to(device), decoder_input_ids=decoder_input_ids.to(device))

            loss = outputs.loss
            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logging.info('Train: {}/{} epoch, {}/{} batch, loss: {}'.format(epoch_idx+1, args.epoch, batch_idx+1, len(train_dataloader), loss))

        logging.info('{}/{} epoch, train loss: {}'.format(epoch_idx, args.epoch, train_loss))

    logging.info('save model to {}...'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


def inference(args, original, simplified, model_save_path, result_save_path, device):
    model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)
    model.load_state_dict(torch.load(model_save_path)) 
    model.eval()
    bart_tokenizer = BartTokenizer.from_pretrained(args.model_name)
    full_dataset = SimplificationDataset(original, simplified, bart_tokenizer, model.prepare_decoder_input_ids_from_labels, args.max_length)
    logging.info('{} data for prediction'.format(len(full_dataset)))
    dataloader = DataLoader(full_dataset, batch_size=20, shuffle=False)
    
    result = {'original': [], 'simplified': []}
    for input_ids, attn_mask, label_ids, decoder_input_ids in tqdm(dataloader):
        output_ids = model.generate(
            inputs=input_ids.to(device),
            attention_mask=attn_mask.to(device),
            num_beams=int(args.num_beams),
            max_length=int(args.max_length),
            # return_dict_in_generate=True # transformers.generation_utils.BeamSearchEncoderDecoderOutput
        )
        original = bart_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = bart_tokenizer.pad_token_id
        simplified = bart_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        result['original'].extend(original)
        result['simplified'].extend(simplified)
    
    pd.DataFrame(result).to_csv(result_save_path, index=False)



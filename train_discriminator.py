import argparse
from random import random
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import ujson

class CodeCompilableDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_source_length) -> None:
        self.code_blocks = self.get_code_from_jsonl(jsonl_file)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
    def get_code_from_jsonl(self, jsonl_file):
        code_blocks = []
        f = open(jsonl_file, "r")
        while True:
            json_content = f.readline()
            if not json_content:
                break
            
            data = ujson.decode(json_content)

            code_blocks.append(data["src_fm"])
        
        f.close()
        return code_blocks

    def __len__(self):
        return len(self.code_blocks)
    
    def swap_pos(self, list, pos1, pos2):
        list[pos1], list[pos2] = list[pos2], list[pos1]
        return list
    def __getitem__(self, index):
        source_str = self.code_blocks[index]
        swap_tokens = False
        # probability of 50% of corrupting the source code
        if random() > 0.5:
            target = 0
            if random() > 0.5:
                # truncate 1/4 of the code source
                source_str = source_str[0:3*len(source_str) // 4]
            else:
                swap_tokens = True
        else:
            target = 1

        input_ids = self.tokenizer.encode(source_str, max_length=self.max_source_length, padding='max_length', truncation=True)
        
        # consider that it was truncated because > max_source_len, thus being incompilable
        if self.tokenizer.pad_token_id not in input_ids:
            target = 0


        if swap_tokens:
            last_non_pad_token_id = input_ids.index(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id in input_ids else len(input_ids) 
            middle_idx = last_non_pad_token_id // 2
            input_ids = self.swap_pos(input_ids, middle_idx, middle_idx + 1)

            
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        mask = ids_tensor.ne(self.tokenizer.pad_token_id)
        
        return {
            'ids': ids_tensor,
            'mask': mask,
            'targets': torch.tensor(target, dtype=torch.float)
        }



def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def optimize_model(model, epoch, training_loader, device, optimizer,  criterion, tb_writer = None):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(tqdm(training_loader)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        outputs = model(ids, mask, labels = targets)

        # loss = criterion(outputs, targets)
        loss = outputs.loss
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.logits, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        # if _%5000==0:
        #     loss_step = tr_loss/nb_tr_steps
        #     accu_step = (n_correct*100)/nb_tr_examples 
        #     print(f"Training Loss per 5000 steps: {loss_step}")
        #     print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    if tb_writer is not None:
        tb_writer.add_scalar('train_loss', epoch_loss, epoch)
        tb_writer.add_scalar('train_acc', epoch_accu, epoch)


def eval(model, val_loader, device, criterion, tb_writer: SummaryWriter = None):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, labels = targets)
            # loss = criterion(outputs, targets)
            loss = outputs.loss
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.logits, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            # if _%5000==0:
            #     loss_step = tr_loss/nb_tr_steps
            #     accu_step = (n_correct*100)/nb_tr_examples
            #     print(f"Validation Loss per 100 steps: {loss_step}")
            #     print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_acc = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_acc}")
    if tb_writer is not None:
        tb_writer.add_scalar('val_loss', epoch_loss, epoch)
        tb_writer.add_scalar('val_locc', epoch_acc, epoch)
    return epoch_acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                         default='E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\\corpus\\json\\train-jsonl\\train100k.jsonl')
    parser.add_argument("--val_file", type=str,
                         default='E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\corpus\json\eval-jsonl\\eval.jsonl')
    parser.add_argument("--output_dir", type=str,
                         default='E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\\corpus\\json\\eval')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=256)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels = 2)

    tb_writer = SummaryWriter(args.output_dir)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)
    
    train_set = CodeCompilableDataset(args.train_file, tokenizer, args.max_source_length)
    print(f"Train set has {len(train_set)} samples")
    val_set = CodeCompilableDataset(args.val_file, tokenizer, args.max_source_length)
    print(f"Val set has {len(val_set)} samples")
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size)

    model.to(device)
    best_val_acc = -1
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        optimize_model(model=model,
                       epoch=epoch,
                       training_loader=train_loader,
                       device=device,
                       optimizer= optimizer,
                       criterion=criterion,
                       tb_writer=tb_writer)
        
        val_acc = eval(model, val_loader, device, criterion, tb_writer)
        if val_acc > best_val_acc:
            print("Best validation accuracy changed to ", val_acc)
            torch.save(model.state_dict(), f"{args.output_dir}/code_discriminator_best_acc.pth")
            best_val_acc = val_acc


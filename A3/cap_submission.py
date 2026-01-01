import pandas as pd
from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from transformers import default_data_collator
# ##########
# TODO: Add more imports
import os
from transformers import AutoProcessor, AutoTokenizer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel
# ##########

class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = 'google/vit-base-patch16-224'
    decoder = 'gpt2'

    # Dataset path
    root_dir = "./flickr8k"

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = "tanishq3"
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 16
    lr = 5e-5
    epochs = 5

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 45     # TODO: Can play around
    weight_decay = 0.001

    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 5
    
class FlickrDataset(Dataset):
    def __init__(
        self,
        args,
        processor,
        tokenizer,
        mode: str = "train",
    ):
        assert mode in ["train", "val", "test"]
        self.args = args
        # ####################
        # TODO: Load Flickr8k dataset
        images = os.path.join(
                                args.root_dir, 
                                "images"
                        )
        captions_file = os.path.join(
                                        args.root_dir, 
                                        f"{mode}.txt"
                                    )

        captions = pd.read_csv(
            captions_file, 
            sep=";",
            header=None, 
            names=["image", "caption"]
        )
        
        self.img_paths = [
            os.path.join(images, row["image"]) 
            for _, row in captions.iterrows()
        ][1::]
        self.captions = captions["caption"].tolist()[1::]

        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer
        self.processor = processor 
        self.tokenizer = tokenizer 
        self. tokenizer.add_special_tokens({
                                            "bos_token": "<|beginoftext|>", 
                                            "pad_token": "<|PAD|>", 
                                            "eos_token": "<|endoftext|>"
                        })
        
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ####################
        # TODO: Load and process image-caption data
        path = self.img_paths[idx]
        captions = self.captions[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        labels = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.args.max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        encoding = {
            "pixel_values": pixel_values,       # Return processed image as a tensor
            "labels": labels,             # Return tokenized caption as a padded tensor
            "path": path,
            "captions": captions,
        }
        # ####################

        return encoding


def train_cap_model(args):
    # Define your vision processor and language tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    processor = AutoProcessor.from_pretrained(args.encoder)

    # Define your Image Captioning model using Vision-Encoder-Decoder model
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder,
        args.decoder,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)    # NOTE: Send your model to GPU

    # Modify the embedding lookup table in decoder model and the tokenizer
    # to include bos_token "<|beginoftext|>" and pad_token "<|pad|>"
    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"
    tokenizer.add_special_tokens({
                                    "bos_token": "<|beginoftext|>", 
                                    "pad_token": "<|PAD|>", 
                                    "eos_token": "<|endoftext|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.resize_token_embeddings(len(tokenizer))


    # Load train/val dataset
    train_dataset = FlickrDataset(
                                args, 
                                tokenizer=tokenizer, 
                                processor=processor, 
                                mode="train"
                )
    val_dataset = FlickrDataset(
                                args, 
                                tokenizer=tokenizer, 
                                processor=processor, 
                                mode="val"
                )

    # Model configuration. 
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # TODO: Play around with some generation config parameters
    # e.g. For beam search, you can potentially have a larger beam size of 5
    # Add more as you see fit
    model.generation_config.max_length = args.max_length #None
    model.generation_config.num_beams = args.num_beams #None
    model.generation_config.do_sample = True
    
    # TODO: Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="bleu_score",
        save_total_limit=1,
        save_strategy="epoch"
    )

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Start training
    # TODO: A good performing model should easily reach a BLEU score above 0.07
    trainer.train()
    trainer.save_model(args.name)
    tokenizer.save_pretrained(args.name)

def load_trained_model(
    ckpt_dir: str,
    ):
    """TODO: Load your best trained model, processor and tokenizer.
    """
    # TODO: Load your model configuration
    config = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)

    # TODO: Load encoder processor
    processor = AutoProcessor.from_pretrained(ckpt_dir)

    # TODO: Load decoder tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    
    # TODO: Load your best trained model
    model = config
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer



def inference(
    img_path,
    model, 
    processor,
    tokenizer,
    ):
    """TODO: Example inference function to predict a caption for an image.
    """
    # TODO: Load and process the image
    image = Image.open(img_path).convert("RGB")
    img_tensor = processor(
                            images = image, 
                            textreturn_tensors = "pt"
                ).pixel_values   # TODO: Preproces the image

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        model.cuda()

    # TODO: Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = model.generate(img_tensor)

    # Tokens -> Str
    generated_caption = tokenizer.decode(
                                generated_ids[0], 
                                skip_special_tokens= True
                        )

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """NOTE: DO NOT CHANGE.
    Compute BLEU score.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }

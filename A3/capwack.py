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
from transformers import AutoProcessor
from transformers import AutoTokenizer
from PIL import Image
from transformers import VisionEncoderDecoderModel
import pandas as pd
from transformers import Seq2SeqTrainingArguments

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoProcessor, AutoTokenizer

# ##########


class Args:
    """Configuration."""

    # Encoder-Decoder for captioning
    encoder = None
    decoder = None

    # Dataset path
    root_dir = "./flickr8k"

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = ""
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 64
    lr = 5e-5
    epochs = 5

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 45  # TODO: Can play around

    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 50


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
        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer
        self.processor = AutoProcessor.from_pretrained("your-vision-encoder-model-name")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "your-language-decoder-model-name"
        )

        # Example paths (you'll need to adjust these)
        images_dir = os.path.join(args.data_dir, "Flickr8k_images")
        captions_file = os.path.join(args.data_dir, "Flickr8k.token.txt")

        # Read captions
        captions_df = pd.read_csv(
            captions_file, delimiter="\t", header=None, names=["image", "caption"]
        )
        self.img_paths = [
            os.path.join(images_dir, row["image"]) for _, row in captions_df.iterrows()
        ]
        self.captions = captions_df["caption"].tolist()

        self.img_paths, self.captions = None, None
        # ####################

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ####################
        # TODO: Load and process image-caption data
        image = Image.open(self.img_paths[idx]).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze()
        caption = self.captions[idx]
        labels = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        ).input_ids.squeeze()
        encoding = {
            "pixel_values": pixel_values,  # Processed image tensor
            "labels": labels,  # Tokenized caption tensor
            "path": self.img_paths[idx],
            "captions": caption,
        }

        encoding = {
            "pixel_values": None,  # Return processed image as a tensor
            "labels": None,  # Return tokenized caption as a padded tensor
            "path": self.img_paths[idx],
            "captions": self.captions[idx],
        }
        # ####################

        return encoding


def train_cap_model(args):
    # Define your vision processor and language tokenizer
    tokenizer = None
    processor = None

    # Define your Image Captioning model using Vision-Encoder-Decoder model
    # Define the Vision-Encoder-Decoder model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "facebook/deit-base-distilled-patch16-224",  # Vision Encoder
        "gpt2",  # Language Decoder
    )

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load train/val dataset
    train_dataset = FlickrDataset(
        args, "train", tokenizer=tokenizer, processor=processor
    )
    val_dataset = FlickrDataset(args, "val", tokenizer=tokenizer, processor=processor)
    # Assuming args contains necessary configurations like data directories
    train_dataset = FlickrDataset(
        args, processor=processor, tokenizer=tokenizer, mode="train"
    )
    val_dataset = FlickrDataset(
        args, processor=processor, tokenizer=tokenizer, mode="val"
    )

    # Model configuration.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Example: For GPT-2, which doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    # TODO: Play around with some generation config parameters
    # e.g. For beam search, you can potentially have a larger beam size of 5
    # Add more as you see fit
    model.generation_config.max_length = args.max_length  # None
    model.generation_config.num_beams = args.num_beams  # None
    # Set default values if not provided
    model.generation_config.max_length = getattr(args, "max_length", 50)
    model.generation_config.num_beams = getattr(args, "num_beams", 5)
    model.generation_config.no_repeat_ngram_size = 2  # Prevents repeating n-grams
    model.generation_config.early_stopping = True

    # TODO: Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    training_args = None
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,  # Directory to save checkpoints
        evaluation_strategy="steps",  # Evaluation strategy
        eval_steps=500,  # Number of steps between evaluations
        per_device_train_batch_size=16,  # Adjust based on GPU memory
        per_device_eval_batch_size=16,  # Adjust based on GPU memory
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
        learning_rate=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay for optimizer
        save_total_limit=3,  # Maximum number of checkpoints to save
        num_train_epochs=10,  # Number of training epochs
        predict_with_generate=True,  # Use generate for evaluation
        logging_dir="./logs",  # Directory for logs
        logging_steps=100,  # Log every 100 steps
        fp16=True,  # Use mixed precision if supported
        save_steps=500,  # Save checkpoint every 500 steps
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="bleu",  # Metric to determine the best model
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


def load_trained_model(
    ckpt_dir: str,
):
    """TODO: Load your best trained model, processor and tokenizer."""
    # TODO: Load your model configuration
    config = None

    # TODO: Load encoder processor
    processor = None
    processor = AutoProcessor.from_pretrained(ckpt_dir)

    # TODO: Load decoder tokenizer
    tokenizer = None
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    # TODO: Load your best trained model
    model = None
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)

    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer


def inference(
    img_path,
    model,
    processor,
    tokenizer,
):
    """TODO: Example inference function to predict a caption for an image."""
    # TODO: Load and process the image
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(img_path).convert("RGB")
    img_tensor = None  # TODO: Preproces the image
    inputs = processor(images=image, return_tensors="pt")
    img_tensor = inputs["pixel_values"]  # Shape: (1, C, H, W)

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # TODO: Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = None
    with torch.no_grad():
        generated_ids = model.generate(
            img_tensor,
            max_length=50,  # Maximum length of the generated caption
            num_beams=5,  # Beam search size
            early_stopping=True,  # Stop early if all beams finish
            no_repeat_ngram_size=2,  # Prevent repeating n-grams
            temperature=1.0,  # Sampling temperature
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Top-K sampling
        )

    # Tokens -> Str
    generated_caption = None
    # Tokens -> Str
    generated_caption = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return generated_caption


def compute_bleu_score(pred, tokenizer):
    """NOTE: DO NOT CHANGE.
    Compute BLEU score.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids  # .squeeze(1)

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

    return {"bleu_score": bleu_score}

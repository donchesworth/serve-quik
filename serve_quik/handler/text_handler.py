# mostly taken from PyTorch's TorchServe github examples at
# examples/Huggingface_Transformers/Transformer_handler_generalized.py
# but added:
#   - nested json to include prob_* for each class
#   - a fourth type of sequence
# sequence_classification. Kept all other functions for future use.

from abc import ABC
import json
import logging
from pathlib import Path
import ast
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

AUTO_MODELS = {
    "sequence_classification": AutoModelForSequenceClassification,
    "sequence_to_sequence": AutoModelForSeq2SeqLM,
    "question_answering": AutoModelForQuestionAnswering,
    "token_classification": AutoModelForTokenClassification,
}


class TextHandler(BaseHandler, ABC):
    """
    Transformers handler class for Sequence classification and
    Seq2Seq models, customized from Huggingface's
    Transformer_handler_generalized.py
    """

    def __init__(self):
        super(TextHandler, self).__init__()
        self.initialized = False

    def encoding(self, input_text):
        max_length = self.setup_config["max_length"]
        # preprocessing text for question_answering.
        if self.setup_config["mode"] == "question_answering":
            question_context = ast.literal_eval(input_text)
            question = question_context["question"]
            context = question_context["context"]
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                max_length=int(max_length),
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        # preprocessing text for classification and seq2seq
        else:
            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        return inputs

    def load_model(self):
        if self.setup_config["save_mode"] == "torchscript":
            model_pt_path = self.model_dir.joinpath(
                self.manifest["model"]["serializedFile"]
            )
            self.model = torch.jit.load(
                model_pt_path, map_location=self.device
                )
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] in AUTO_MODELS:                
                auto_model = AUTO_MODELS[self.setup_config["mode"]]
                logger.info(f"auto_model is string: {str(auto_model)}")
                logger.info(f"auto_model is type: {type(auto_model)}")
                logger.info(f"auto_model is: {print(auto_model)}")
                self.model = auto_model.from_pretrained(self.model_dir)
            else:
                logger.warning("Missing the operation mode.")
            self.model.to(self.device)
        else:
            logger.warning("Missing the checkpoint or state_dict.")

    def json_load(self, filename):
        filepath = self.model_dir.joinpath(filename)
        if filepath.is_file():
            with open(filepath) as file:
                config = json.load(file)
            return config
        else:
            logger.warning(f"Missing the {filename} file.")
            return None

    def classify_token(self, input_batch):
        inferences = []
        outputs = self.model(*input_batch)[0]
        logger.info(
            "The token classification model size is ",
            outputs.size(),
        )
        logger.info("The output from the Seq classification model: ", outputs)
        num_rows = outputs.shape[0]
        for i in range(num_rows):
            output = outputs[i].unsqueeze(0)
            predictions = torch.argmax(output, dim=2)
            decoded = self.tokenizer.decode(input_batch[0][i])
            tokens = self.tokenizer.tokenize(decoded)
            if self.mapping:
                label_list = self.mapping["label_list"]
            label_list = label_list.strip("][").split(", ")
            prediction = [
                (token, label_list[prediction])
                for token, prediction in zip(tokens, predictions[0].tolist())
            ]
            inferences.append(prediction)
        logger.info(f"Model predicted: '{prediction}'")
        return inferences

    def classify_sequence(self, input_batch):
        inferences = []
        predictions = self.model(*input_batch)
        logger.info(
            "The Seq classification model size is ",
            predictions[0].size(),
        )
        logger.info(
            "The output from the Seq classification model: ",
            predictions,
        )
        num_rows, num_cols = predictions[0].shape
        prob_keys = list(map(str.lower, self.mapping.values()))
        for i in range(num_rows):
            logits = predictions[0][i].unsqueeze(0)
            y_hat = logits.argmax(1).item()
            predicted_idx = str(y_hat)
            probs = F.softmax(logits, dim=1)
            probs = probs.detach().numpy().tolist()[0]
            probs = dict(zip(prob_keys, probs))
            labels = self.mapping[predicted_idx]
            inferences.append({"prob": probs, "label": labels})
        return inferences

    def answer_question(self, input_batch):
        # the output should be only answer_start and answer_end
        # we are outputing the words just for demonstration.
        inferences = []
        if self.setup_config["save_mode"] == "pretrained":
            outputs = self.model(input_batch)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
        else:
            answer_start_scores, answer_end_scores = self.model(*input_batch)
        logger.info(
            "Q&A answer start scores size is ",
            answer_start_scores.size(),
        )
        logger.info(
            "The output for Q&A answer start scores: ",
            answer_start_scores,
        )
        logger.info(
            "Q&A answer end scores size is ",
            answer_end_scores.size(),
        )
        logger.info(
            "The output for Q&A answer start scores: ",
            answer_end_scores,
        )

        num_rows, num_cols = answer_start_scores.shape
        # inferences = []
        for i in range(num_rows):
            answer_start_scores_one_seq = answer_start_scores[i].unsqueeze(0)
            answer_start = torch.argmax(answer_start_scores_one_seq)
            answer_end_scores_one_seq = answer_end_scores[i].unsqueeze(0)
            answer_end = torch.argmax(answer_end_scores_one_seq) + 1
            prediction = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    input_batch[0][i].tolist()[answer_start:answer_end]
                )
            )
            inferences.append(prediction)
        logger.info(f"Model predicted: '{prediction}'")
        return inferences

    def initialize(self, ctx):
        """In this initialize function, the transformers model is loaded
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.model_dir = Path(properties.get("model_dir"))

        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            dev = "cuda:" + str(properties.get("gpu_id"))
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        # read configs for the mode, model_name, etc. from setup_config.json
        self.setup_config = self.json_load("setup_config.json")

        # Loading the model and tokenizer from checkpoint and config files
        # based on the user's choice of mode further setup config can be added.
        self.load_model()

        if any(list(self.model_dir.glob("vocab.*"))):
            model_name_or_path = self.model_dir
        else:
            model_name_or_path = self.setup_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=self.setup_config["do_lower_case"]
        )

        self.model.eval()
        logger.info(f"Model from path {self.model_dir} loaded successfully")

        # Classification needs index to name
        if self.setup_config["mode"] in [
            "sequence_classification",
            "token_classification",
        ]:
            self.mapping = self.json_load("index_to_name.json")
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the choice of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on
            to the preprocess function.
        Returns:
            list : The preprocess function returns a list of Tensor for the
            size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            logger.info(f"Received text: '{input_text}'")
            inputs = self.encoding(input_text)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for padded input tokens.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((
                        input_ids_batch, input_ids), 0
                    )
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process
            function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        inferences = []
        # Handling inference for sequence_classification.
        if self.setup_config["mode"] == "sequence_classification":
            inferences = self.classify_sequence(input_batch)
        # Handling inference for question_answering.
        elif self.setup_config["mode"] == "question_answering":
            inferences = self.answer_question(input_batch)
        # Handling inference for token_classification.
        elif self.setup_config["mode"] == "token_classification":
            inferences = self.classify_tokens(input_batch)
        elif self.setup_config["mode"] == "sequence_to_sequence":
            input_ids_batch, attention_mask_batch = input_batch
            gen = self.model.generate(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch
            )
            prediction = self.tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )
            inferences.append(prediction)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into
        Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of
            the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

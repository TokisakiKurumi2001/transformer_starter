import argparse
import os
import torch
import simple_transformer as T
from typing import List
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from simple_transformer.data import Vocab


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(description='Transformer testing')
    parser.add_argument('checkpoint_path', type=str, help='Checkpoint path')
    parser.add_argument('--config_path', type=str, default='config/translator.beam.yaml', help='Translator config path')
    parser.add_argument('--overrides', type=str, default=None, help='Overrides for translator config') # "{'beam_size':2}"
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    # Load config and vocab pair
    model_dir = os.path.dirname(args.checkpoint_path)
    config = T.load_config(os.path.join(model_dir, 'config.yaml'))
    source_vocab, target_vocab = T.load_vocab_pair(**config.vocab)

    # Load a pretrained transformer
    model = T.make_model(
        input_vocab_size= len(source_vocab),
        output_vocab_size=len(target_vocab),
        **config.model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Build translator
    translator_config = T.load_config(args.config_path)
    if args.overrides is not None:
        translator_config = translator_config.update(eval(args.overrides))
    print(f'Translator:\n{translator_config}')
    translator = T.make_translator(
        model=model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        **translator_config)

    # Load test dataset and translate
    test_dataset = T.load_dataset(split='test', **config.dataset)
    outputs = []
    targets = []
    self_outputs = []
    sources = []
    for source_text, target_text in tqdm(test_dataset):
        output, self_output = translator(source_text)
        target = [target_vocab.tokenize(target_text)]
        source = [source_vocab.tokenize(source_text)]
        outputs.append(output)
        self_outputs.append(self_output)
        targets.append(target)
        sources.append(source)

        if args.verbose:
            sentence = handle_spaces(output, target_vocab)
            print('-'*100)
            print(source_text + target_text + sentence)

    # Compute BLEU score
    score = bleu_score(outputs, targets)
    print(f'BLEU score: {score}')
    self_score = bleu_score(self_outputs, sources)
    print(f'Self BLEU score: {self_score}')


def handle_spaces(output: List[str], target_vocab: Vocab) -> str:
    # Simple handling of spaces (not the best)
    sentence = ''
    for token in output:
        if len(sentence)==0 or token in ('.', '!', ',', ';', ':', '\''):
            sentence += token
        else:
            sentence += ' ' + token
    return sentence


if __name__=='__main__':
    main()

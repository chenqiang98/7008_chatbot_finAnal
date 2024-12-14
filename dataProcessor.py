import pandas as pd
import os
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Text(object):
    """A single train/dev/test example for simple sequence classification."""

    def __init__(self, text, label=None):
        self.text = text
        self.label = label


class TextFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `Text`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `Text`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `Text`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # dicts = []
        data = pd.read_csv(input_file)
        return data


class SentiAnalDataProcessor(DataProcessor):
    """
    Returns:
        examples: dataset，including index, text and class
    """
    def __init__(self):
        self.labels = ['inadequate & non-urgent', 
                       'inadequate & urgent', 
                       'adequate & non-urgent',
                       'adequate & urgent',]

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'train.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'dev.csv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'test.csv')), 'test')
    
    def get_single_test_example(self, text):
        return [Text(text=text, label=None)]

    def get_labels(self):
        return self.labels

    def _create_examples(self, data, set_type):
        examples = []
        max_len = 0
        for index, row in data.iterrows():
            text = str(row['desc'])
            max_len = max(max_len, len(text.split()))
            label = f"{row['reason_adequacy']} & {row['urgency']}" if set_type != 'test' else None
            examples.append(Text(text=text, label=label))
        # print(max_len)
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=False):
    """Loads a data file into a list of `InputBatch`s.

    Args:
        examples      : [List] input sample，sentence and label
        label_list    : [List] labels of all classes
        max_seq_length: [int] max length of text
        tokenizer     : [Method] method for tokenization

    Returns:
        features:
            input_ids  : [ListOf] id of token (each id corresponding to a word vector)
            input_mask : [ListOfInt] Word=1, Pad=0
            segment_ids: [ListOfInt] Sentence identifiers: the first sentence is all 0s, the second sentence is all 1s (currently inactive, all 0s)
            label_id   : [ListOfInt] Convert Label_list to the corresponding id representation
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        encode_dict = tokenizer.encode_plus(text=example.text,
                                            add_special_tokens=True,
                                            max_length=max_seq_length,
                                            padding='max_length',
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            truncation=True)

        input_ids = encode_dict['input_ids']
        input_mask = encode_dict['attention_mask']
        segment_ids = encode_dict['token_type_ids']

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label] if example.label is not None else -1
        # if label_id < 0 or label_id >= len(label_list):
        #     print(example.label, example.text)
        #     assert 0
        if ex_index % 50000 == 0 and show_exp:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % example.text)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            TextFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

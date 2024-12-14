
import argparse

from torch.utils.data import DataLoader
from dataset import *
from dataProcessor import *
import time
from transformers import BertTokenizer
from transformers import logging as lg
import logging

def sentiment_analysis(text):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default="")
    # args = parser.parse_args()
    # assert args.model_path != ""


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    lg.set_verbosity_warning()
    data_dir = "."
    bert_dir = "bert-en"
    output_path = "submission.txt"
    model_path = "selected_epoch.pth"
    my_processor = SentiAnalDataProcessor()
    label_list = my_processor.get_labels()

    # ---------------
    # For batch data
    # test_data = my_processor.get_test_examples(data_dir)
    # ---------------
    # For single test data
    # text = "Student loan, to pay for tuition"
    test_data = my_processor.get_single_test_example(text)
    # ---------------

    tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=True)

    test_features = convert_examples_to_features(test_data, label_list, 64, tokenizer)
    test_dataset = SentiAnalDataset(test_features, 'test')
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=64)

    test_data_len = len(test_dataset)
    logger.info(f"Length of test dataset：{test_data_len}")


    my_model = torch.load(model_path, map_location='cpu')
    my_model = my_model.to(device)

    start_time = time.time()
    my_model.eval()

    # fout = open(output_path, "w", encoding='utf-8')
    # for step, batch_data in enumerate(test_data_loader):
    #     input_ids = batch_data['input_ids'].to(device)
    #     input_mask = batch_data['input_mask'].to(device)
    #     segment_ids = batch_data['segment_ids'].to(device)
    #     output = my_model(input_ids, input_mask, segment_ids)
    #     result = output.argmax(1)
    #     for i in result:
    #         fout.write(f"{label_list[i]}\n")

    for batch_data in test_data_loader:
        input_ids = batch_data['input_ids'].to(device)
        input_mask = batch_data['input_mask'].to(device)
        segment_ids = batch_data['segment_ids'].to(device)
        output = my_model(input_ids, input_mask, segment_ids)
        result = output.argmax(1)
        return label_list[result]

    # end_time = time.time()
    # logger.info(f'Testing time: {(end_time - start_time) / 3600}h')


if __name__ == '__main__':
    print(sentiment_analysis("Student loan, to pay for tuition"))
